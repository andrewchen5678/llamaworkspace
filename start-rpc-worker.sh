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

# Preflight: the Linux CI binary links libgomp (OpenMP); minimal installs lack
# it, so exec'ing would fail with a cryptic "libgomp.so.1: cannot open shared
# object file". Catch any missing shared lib here with a fix-it hint instead.
if command -v ldd >/dev/null 2>&1; then
  missing="$(ldd "$RPC_SERVER" 2>/dev/null | awk '/not found/ {print $1}')"
  if [[ -n "$missing" ]]; then
    err "rpc-server is missing shared libraries:"
    printf '       %s\n' $missing >&2
    err "Install the OpenMP runtime your distro ships it in:"
    err "  Debian/Ubuntu : sudo apt-get install -y libgomp1"
    err "  Fedora/RHEL   : sudo dnf install -y libgomp"
    err "  Alpine        : sudo apk add libgomp"
    exit 1
  fi
fi

log "Starting RPC worker on ${RPC_HOST}:${RPC_PORT}"
log "Binary: $RPC_SERVER"
log "Reminder: trusted LAN/VPN only — this endpoint has no authentication."
echo

# -c enables a local tensor cache (~/.cache/llama.cpp/rpc) so repeated loads of
# the same weights are fast.
exec "$RPC_SERVER" -c -H "$RPC_HOST" -p "$RPC_PORT"
