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
#   TENSOR_SPLIT  (optional) split weights e.g. "32,32,16" = [local, w1, w2] by RAM
#   PORT          listen port            (default: 8090)
#   HOST          bind address           (default: 127.0.0.1)
#   CTX           context size           (default: 16384)
#   MODEL         model path             (default: ./models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf)
#   LLAMACPP_DIR  dir holding llama-server (default: ./llama.cpp/bin)

set -euo pipefail

# Run from this script's directory so the default relative paths resolve.
cd "$(dirname "$0")"

# ---- configuration ---------------------------------------------------------
PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-16384}"
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

# The CI binaries ship their shared libs (libggml.so.0, libllama.so, …) right
# next to llama-server, but the linker doesn't search the binary's own directory.
# Point the loader at the lib dir (absolute path so it survives `exec`).
LIB_DIR="$(cd "$LLAMACPP_DIR" && pwd)"
export LD_LIBRARY_PATH="${LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DYLD_LIBRARY_PATH="${LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

# Qwen3.5-35B-A3B recommended general sampling (thinking ON, generic use).
SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5)

# --tensor-split is optional; only pass it when TENSOR_SPLIT is set.
SPLIT=()
if [[ -n "${TENSOR_SPLIT:-}" ]]; then
  SPLIT=(--tensor-split "$TENSOR_SPLIT")
fi

# ---- main ------------------------------------------------------------------
log "Model       : $MODEL"
log "Workers     : $RPC_WORKERS"
log "Tensor split: ${TENSOR_SPLIT:-(auto)}"
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
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --rpc "$RPC_WORKERS" \
  "${SPLIT[@]}" \
  -ngl 999 \
  -c "$CTX" \
  --jinja \
  "${SAMPLING[@]}" \
  --alias qwen3.5-35b-a3b-cluster \
  --host "$HOST" \
  --port "$PORT"
