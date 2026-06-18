#!/usr/bin/env bash
#
# serve-qwen3-cluster.sh
#
# Launches Qwen3.5-35B-A3B split across the cluster via llama.cpp RPC. Run this on
# the MAIN node (the Mac). It starts llama-server locally and offloads part of
# the model to the worker rpc-servers named in RPC_WORKERS.
#
# This is deliberately SEPARATE from llama-swap. llama-swap handles the
# single-machine Gemma models; this is a standalone, deliberately-spun-up cluster
# service with no idle auto-unload — it runs until you stop it (Ctrl-C).
#
# It listens on port 8090, the SAME port as llama-swap. Run only ONE at a time:
# stop llama-swap before starting the cluster, and vice versa.
#
# Start the workers first (on each non-Mac box):  ./start-rpc-worker.sh
#
# Usage:
#   RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 ./serve-qwen3-cluster.sh
#
# Environment:
#   RPC_WORKERS   (required) comma-separated worker endpoints (ip:port,ip:port)
#   TENSOR_SPLIT  (optional) e.g. "45,45,10" = [w1, w2, local] — local is LAST
#   NO_MMAP       1 = pass --no-mmap; frees the main node's RAM (default: unset)
#   PORT          listen port            (default: 8090)
#   HOST          bind address           (default: 127.0.0.1)
#   CTX           context size           (default: 262144 — Qwen3.5's native max)
#   MODEL         model path             (default: ./models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf)
#   LLAMACPP_DIR  dir holding llama-server (default: ./llama.cpp/bin)

set -euo pipefail

# Run from this script's directory so the default relative paths resolve.
cd "$(dirname "$0")"

# ---- configuration ---------------------------------------------------------
PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"
# Qwen3.5-35B-A3B's native context length (262144 = 256K). Extensible to ~1M with
# YaRN, but that needs explicit RoPE-scaling flags; the native max is the ceiling
# here. The KV cache for 256K is large in aggregate, but llama.cpp allocates it
# per layer on the node holding that layer, so it's split across the cluster (not
# duplicated) — bias --tensor-split toward roomier nodes, or lower CTX, if a node
# runs out of memory.
CTX="${CTX:-262144}"
MODEL="${MODEL:-./models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
LLAMACPP_DIR="${LLAMACPP_DIR:-./llama.cpp/bin}"
LLAMA_SERVER="${LLAMACPP_DIR}/llama-server"

# ---- helpers ---------------------------------------------------------------
log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# ---- preflight -------------------------------------------------------------
if [[ -z "${RPC_WORKERS:-}" ]]; then
  err "RPC_WORKERS is required — comma-separated worker endpoints."
  err "Example: RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 ./serve-qwen3-cluster.sh"
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
  err "Download it first:  ./download-qwen3.5-35b-a3b.sh"
  exit 1
fi

# Qwen3.5-35B-A3B recommended general sampling (thinking ON, generic use).
SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5)

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
# (~21 GB) no matter how little --tensor-split leaves it to compute — that mapping
# is reclaimable file cache, but it dominates "Memory Used". --no-mmap instead
# reads weights into buffers and streams the remote layers through a staging
# buffer, so only the local layer share stays resident (measured: 20.8 GB -> 3.3 GB
# RSS at 45,45,10). Cost: the full file is read from disk at load (~25s slower
# here) and there's no cross-restart page cache. Worth it to keep a workstation
# main node light; skip it if the main node has RAM to spare and you restart often.
MMAP=()
if [[ "${NO_MMAP:-}" == "1" ]]; then
  MMAP=(--no-mmap)
fi

# ---- main ------------------------------------------------------------------
log "Model       : $MODEL"
log "Workers     : $RPC_WORKERS"
log "Tensor split: ${TENSOR_SPLIT:-(auto)}"
log "mmap        : $([[ "${NO_MMAP:-}" == "1" ]] && echo 'off (--no-mmap, frees main-node RAM)' || echo 'on')"
log "Listening   : http://${HOST}:${PORT}  (alias: qwen3.5-35b-a3b-cluster)"
log "Note        : shares port ${PORT} with llama-swap — run only one at a time."
echo

# No --predict / generation cap on purpose (see CLAUDE.md): a cap silently
# truncates; uncapped, the only ceiling is -c, which fails loudly on overflow.
#
# No --context-shift on purpose: Qwen3.5's Gated-DeltaNet layers carry a
# fixed-size recurrent state that can't be partially rewound, so dropping oldest
# tokens would corrupt it. llama.cpp leaves context-shift OFF by default; keep it.
#
# --jinja uses the GGUF's embedded chat template, which drives Qwen3.5's
# thinking/non-thinking split and tool-call parsing (and enables future
# --chat-template-kwargs toggling).
# -sm layer (the default, made explicit) splits the model layer-wise across the
# local device + every --rpc worker, and allocates each layer's KV cache on the
# node that holds it — so the KV cache is distributed across the cluster, not
# duplicated. --tensor-split sets the proportions (see ORDER note above); omitted,
# it splits proportionally to each device's free memory.
#
# -fit off pins placement to exactly our --tensor-split / -ngl / -c, with no
# auto-fit second-guessing. Note this is belt-and-suspenders, not load-bearing:
# testing showed -fit on vs off produce byte-identical placement once
# --tensor-split and -c are set, so it's purely for determinism. (Either way, the
# "fitting params to device memory ..." log line is followed by a ~40s weight-load
# pause — that's normal loading, not a hang.)
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --rpc "$RPC_WORKERS" \
  --split-mode layer \
  -fit off \
  "${SPLIT[@]}" \
  "${MMAP[@]}" \
  -ngl 999 \
  -c "$CTX" \
  --jinja \
  "${SAMPLING[@]}" \
  --alias qwen3.5-35b-a3b-cluster \
  --host "$HOST" \
  --port "$PORT"
