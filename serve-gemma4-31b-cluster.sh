#!/usr/bin/env bash
#
# serve-gemma4-31b-cluster.sh
#
# Launches Gemma 4 31B (dense) split across the cluster via llama.cpp RPC, with
# MTP speculative decoding. Run this on the MAIN node (the Mac). It starts
# llama-server locally and offloads part of the target model to the worker
# rpc-servers named in RPC_WORKERS; the MTP drafter runs locally.
#
# This is deliberately SEPARATE from llama-swap (which handles the single-machine
# Gemma E4B models) and from serve-qwen3-cluster.sh — it's a standalone,
# deliberately-spun-up cluster service with no idle auto-unload. It runs until
# you stop it (Ctrl-C).
#
# It listens on port 8090, the SAME port as llama-swap and the Qwen cluster. Run
# only ONE at a time.
#
# Start the workers first (on each non-Mac box):  ./start-rpc-worker.sh
#
# Usage:
#   RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 ./serve-gemma4-31b-cluster.sh
#
# Environment:
#   RPC_WORKERS   (required) comma-separated worker endpoints (ip:port,ip:port)
#   TENSOR_SPLIT  (optional) e.g. "45,45,10" = [w1, w2, local] — local is LAST
#   NO_MMAP       1 = pass --no-mmap; frees the main node's RAM (default: unset)
#   NO_MTP        1 = disable the MTP drafter (fallback if MTP+RPC misbehaves)
#   SPEC_N_MAX    max tokens the drafter proposes per step  (default: 4)
#   PORT          listen port            (default: 8090)
#   HOST          bind address           (default: 127.0.0.1)
#   CTX           context size           (default: 262144 — Gemma 4 31B's native max)
#   MODEL         target model path      (default: ./models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf)
#   MODEL_DRAFT   MTP drafter path       (default: ./models/mtp-gemma-4-31B-it.gguf)
#   LLAMACPP_DIR  dir holding llama-server (default: ./llama.cpp/bin)

set -euo pipefail

# Run from this script's directory so the default relative paths resolve.
cd "$(dirname "$0")"

# ---- configuration ---------------------------------------------------------
PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"
# Gemma 4 31B's native context length (262144 = 256K). The KV cache for 256K is
# large in aggregate (more so with -fa off, below), but llama.cpp allocates it
# per layer on the node holding that layer, so it's split across the cluster (not
# duplicated) — bias --tensor-split toward roomier nodes, lower CTX, or turn on
# flash attention (-fa on) to shrink it, if a node runs out of memory.
CTX="${CTX:-262144}"
SPEC_N_MAX="${SPEC_N_MAX:-4}"
MODEL="${MODEL:-./models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf}"
MODEL_DRAFT="${MODEL_DRAFT:-./models/mtp-gemma-4-31B-it.gguf}"
LLAMACPP_DIR="${LLAMACPP_DIR:-./llama.cpp/bin}"
LLAMA_SERVER="${LLAMACPP_DIR}/llama-server"

# ---- helpers ---------------------------------------------------------------
log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# ---- preflight -------------------------------------------------------------
if [[ -z "${RPC_WORKERS:-}" ]]; then
  err "RPC_WORKERS is required — comma-separated worker endpoints."
  err "Example: RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 ./serve-gemma4-31b-cluster.sh"
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
  err "Download it first:  ./download-gemma4-31b-mtp.sh"
  exit 1
fi

# MTP is on by default. NO_MTP=1 drops the drafter entirely — use it as a
# fallback if MTP + RPC together crash or stall (not yet validated upstream), to
# confirm the bare split works before reintroducing speculative decoding.
SPEC=()
if [[ "${NO_MTP:-}" != "1" ]]; then
  if [[ ! -f "$MODEL_DRAFT" ]]; then
    err "MTP drafter not found at $MODEL_DRAFT"
    err "Download it first:  ./download-gemma4-31b-mtp.sh   (or set NO_MTP=1)"
    exit 1
  fi
  SPEC=(--model-draft "$MODEL_DRAFT" --spec-type draft-mtp --spec-draft-n-max "$SPEC_N_MAX")
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
SPLIT=()
if [[ -n "${TENSOR_SPLIT:-}" ]]; then
  SPLIT=(--tensor-split "$TENSOR_SPLIT")
fi

# NO_MMAP=1 passes --no-mmap. By default the main node mmaps the ENTIRE GGUF to
# read and stream weights to the workers, so its RSS shows ~the full model size
# (~19 GB) no matter how little --tensor-split leaves it to compute — that mapping
# is reclaimable file cache, but it dominates "Memory Used". --no-mmap instead
# reads weights into buffers and streams the remote layers through a staging
# buffer, so only the local layer share stays resident. Cost: the full file is
# read from disk at load (slower) and there's no cross-restart page cache. Worth
# it to keep a workstation main node light; skip it if the main node has RAM to
# spare and you restart often.
MMAP=()
if [[ "${NO_MMAP:-}" == "1" ]]; then
  MMAP=(--no-mmap)
fi

# ---- main ------------------------------------------------------------------
log "Model       : $MODEL"
log "MTP drafter : $([[ "${NO_MTP:-}" == "1" ]] && echo '(disabled, NO_MTP=1)' || echo "$MODEL_DRAFT (n-max $SPEC_N_MAX)")"
log "Workers     : $RPC_WORKERS"
log "Tensor split: ${TENSOR_SPLIT:-(auto)}"
log "mmap        : $([[ "${NO_MMAP:-}" == "1" ]] && echo 'off (--no-mmap, frees main-node RAM)' || echo 'on')"
log "Listening   : http://${HOST}:${PORT}  (alias: gemma-4-31b-cluster)"
log "Note        : shares port ${PORT} with llama-swap and the Qwen cluster — run only one at a time."
echo

# No --predict / generation cap on purpose (see CLAUDE.md): a cap silently
# truncates; uncapped, the only ceiling is -c, which fails loudly on overflow.
#
# MTP speculative decoding (--spec-type draft-mtp): the drafter proposes tokens,
# the 31B target verifies them, so output is byte-identical to running without it
# — just faster. The drafter runs locally; only the target model splits over RPC.
#
# -fa off matches the working single-machine Gemma config (the E4B llama-swap
# entries). Flash attention shrinks the KV cache, so for very long contexts you
# can try -fa on to free memory — verify MTP acceptance and output are unchanged.
#
# --jinja uses the GGUF's embedded chat template, which drives Gemma 4's
# thinking/non-thinking split and tool-call parsing.
#
# -sm layer (the default, made explicit) splits the target layer-wise across the
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
  "${SPEC[@]}" \
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
  --alias gemma-4-31b-cluster \
  --host "$HOST" \
  --port "$PORT"
