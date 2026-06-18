#!/usr/bin/env bash
#
# start-rpc-worker.sh
#
# Starts the llama.cpp RPC worker on a cluster node. Run this on every WORKER
# (the non-Mac boxes). The main node's serve-qwen3-cluster.sh connects to these
# workers and offloads part of the model to them.
#
# ┌──────────── SECURITY ─────────────────────────────────────────────────────┐
# │ The RPC server has NO authentication and is documented as insecure. Bind it │
# │ only on a trusted LAN or VPN. NEVER expose it to the public internet.        │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   ./start-rpc-worker.sh
#
# Environment overrides:
#   RPC_PORT      port to listen on            (default: 50052)
#   RPC_HOST      bind address                 (default: 0.0.0.0 — LAN-reachable)
#   LLAMACPP_DIR  dir holding rpc-server       (default: ./llama.cpp/bin)

set -euo pipefail

# Run from this script's directory so the default relative paths resolve.
cd "$(dirname "$0")"

# ---- configuration ---------------------------------------------------------
RPC_PORT="${RPC_PORT:-50052}"
RPC_HOST="${RPC_HOST:-0.0.0.0}"
LLAMACPP_DIR="${LLAMACPP_DIR:-./llama.cpp/bin}"
RPC_SERVER="${LLAMACPP_DIR}/rpc-server"

# ---- helpers ---------------------------------------------------------------
log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# ---- main ------------------------------------------------------------------
if [[ ! -x "$RPC_SERVER" ]]; then
  err "rpc-server not found at $RPC_SERVER"
  err "Fetch the pinned build first:  ./fetch-llamacpp-rpc.sh"
  exit 1
fi

log "Starting RPC worker on ${RPC_HOST}:${RPC_PORT}"
log "Binary: $RPC_SERVER"
log "Reminder: trusted LAN/VPN only — this endpoint has no authentication."
echo

# -c enables a local tensor cache (~/.cache/llama.cpp/rpc) so repeated loads of
# the same weights are fast.
exec "$RPC_SERVER" -c -H "$RPC_HOST" -p "$RPC_PORT"
