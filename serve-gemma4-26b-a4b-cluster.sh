#!/usr/bin/env bash
#
# serve-gemma4-26b-a4b-cluster.sh
#
# Launches Gemma 4 26B-A4B (sparse MoE) split across the cluster via llama.cpp
# RPC. Run this on the MAIN node (the Mac). It starts llama-server locally and
# offloads part of the model to the worker rpc-servers named in RPC_WORKERS.
#
# NO MTP: on the MoE the ~4B active params leave little for speculative decoding
# to win back (minimal improvement vs >2x on the dense 31B), so it's run plain.
#
# This is deliberately SEPARATE from llama-swap and from the other cluster serve
# scripts — a standalone, deliberately-spun-up service with no idle auto-unload.
# It runs until you stop it (Ctrl-C).
#
# It listens on port 8090, the SAME port as llama-swap and the other clusters.
# Run only ONE at a time.
#
# Start the workers first (on each non-Mac box):  ./start-rpc-worker.sh
#
# Usage:
#   RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 ./serve-gemma4-26b-a4b-cluster.sh
#
# Environment:
#   RPC_WORKERS   (required) comma-separated worker endpoints (ip:port,ip:port)
#   TENSOR_SPLIT  (optional) e.g. "45,45,10" = [w1, w2, local] — local is LAST.
#                 Setting it also enables --no-mmap (frees the main node's RAM).
#   PORT          listen port            (default: 8090)
#   HOST          bind address           (default: 127.0.0.1)
#   CTX           context size           (default: 262144 — Gemma 4's native max)
#   MODEL         model path             (default: ./models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf)
#   LLAMACPP_DIR  dir holding llama-server (default: ./llama.cpp/bin)

set -euo pipefail

# Run from this script's directory so the default relative paths resolve.
cd "$(dirname "$0")"

# ---- configuration ---------------------------------------------------------
PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"
# Gemma 4 26B-A4B's native context length (262144 = 256K). The KV cache for 256K
# is large in aggregate; llama.cpp allocates it per layer on the node holding that
# layer, so it's split across the cluster (not duplicated) — but a worker with
# little free RAM can still OOM during KV allocation (seen as "Remote RPC server
# crashed"). If that happens, lower CTX (e.g. 32768) or bias --tensor-split toward
# roomier nodes.
CTX="${CTX:-262144}"
MODEL="${MODEL:-./models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
LLAMACPP_DIR="${LLAMACPP_DIR:-./llama.cpp/bin}"
LLAMA_SERVER="${LLAMACPP_DIR}/llama-server"

# ---- helpers ---------------------------------------------------------------
log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# ---- preflight -------------------------------------------------------------
if [[ -z "${RPC_WORKERS:-}" ]]; then
  err "RPC_WORKERS is required — comma-separated worker endpoints."
  err "Example: RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 ./serve-gemma4-26b-a4b-cluster.sh"
  err "Start the workers first with ./start-rpc-worker.sh on each non-Mac box."
  exit 1
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  err "llama-server not found at $LLAMA_SERVER"
  err "Fetch the pinned build first:  ./fetch-llamacpp-rpc.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  err "Model not found at $MODEL"
  err "Download it first:  ./download-gemma4-26b-a4b.sh"
  exit 1
fi

# Gemma's recommended general-purpose sampling. Thinking is left ON (reasoning
# shares the generation budget — see CLAUDE.md). For deterministic summarization,
# override per request (temp 0 / top-k 1) or relaunch with tighter values.
SAMPLING=(--temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0)

# --tensor-split is optional; only pass it when TENSOR_SPLIT is set. In layer
# split mode its proportions govern BOTH the layer weights and the per-layer KV
# cache, so it's the single lever for biasing memory away from a tight node.
#
# ORDER (verified empirically on the b9701 build, NOT [local-first] as you might
# expect): the values map to [each --rpc worker in RPC_WORKERS order, ..., then
# the LOCAL device LAST]. So with two workers, TENSOR_SPLIT=worker1,worker2,mac.
# To keep this Mac light, its share is the LAST value (e.g. 45,45,10 → ~10% local).
# Setting TENSOR_SPLIT also turns on --no-mmap. By default the main node mmaps the
# ENTIRE GGUF to read and stream weights to the workers, so its RSS shows ~the full
# model size (~16 GB) no matter how little --tensor-split leaves it to compute —
# that mapping is reclaimable file cache, but it dominates "Memory Used". --no-mmap
# instead reads weights into buffers and streams the remote layers through a staging
# buffer, so only the local layer share stays resident. Cost: the full file is read
# from disk at load (slower) and there's no cross-restart page cache. The whole point
# of biasing --tensor-split toward the workers is to keep this Mac light, so the two
# go hand in hand: if you've set a split you want the RAM savings too. Without
# TENSOR_SPLIT, mmap stays on.
SPLIT=()
MMAP=()
if [[ -n "${TENSOR_SPLIT:-}" ]]; then
  SPLIT=(--tensor-split "$TENSOR_SPLIT")
  MMAP=(--no-mmap)
fi

# ---- main ------------------------------------------------------------------
log "Model       : $MODEL"
log "Workers     : $RPC_WORKERS"
log "Tensor split: ${TENSOR_SPLIT:-(auto)}"
log "mmap        : $([[ -n "${TENSOR_SPLIT:-}" ]] && echo 'off (--no-mmap, frees main-node RAM)' || echo 'on')"
log "Context     : $CTX"
log "Listening   : http://${HOST}:${PORT}  (alias: gemma-4-26b-a4b-cluster)"
log "Note        : shares port ${PORT} with llama-swap and the other clusters — run only one at a time."
echo

# No --predict / generation cap on purpose (see CLAUDE.md): a cap silently
# truncates; uncapped, the only ceiling is -c, which fails loudly on overflow.
#
# -fa off matches the working single-machine Gemma config (the E4B llama-swap
# entries). Flash attention shrinks the KV cache, so for very long contexts you
# can try -fa on to free memory — verify output is unchanged.
#
# --jinja uses the GGUF's embedded chat template, which drives Gemma 4's
# thinking/non-thinking split and tool-call parsing.
#
# -sm layer (the default, made explicit) splits the model layer-wise across the
# local device + every --rpc worker, allocating each layer's KV cache on the node
# that holds it — so the KV cache is distributed, not duplicated. --tensor-split
# sets the proportions (see ORDER note above); omitted, it splits proportionally
# to each device's free memory.
#
# -fit off pins placement to exactly our --tensor-split / -ngl / -c, with no
# auto-fit second-guessing (purely for determinism; the "fitting params to device
# memory ..." log line is followed by a weight-load pause — normal, not a hang).
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --rpc "$RPC_WORKERS" \
  --split-mode layer \
  -fit off \
  "${SPLIT[@]}" \
  "${MMAP[@]}" \
  -ngl 999 \
  -fa off \
  -c "$CTX" \
  --jinja \
  "${SAMPLING[@]}" \
  --alias gemma-4-26b-a4b-cluster \
  --host "$HOST" \
  --port "$PORT"
