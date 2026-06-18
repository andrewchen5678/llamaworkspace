#!/usr/bin/env bash
# Launch llama-swap in front of llama-server.
#
# llama-swap starts the model on demand and unloads it after the `ttl` in
# llama-swap.yaml (default 5 min idle), freeing GPU/RAM. Requests reload it
# transparently.
#
# Usage:
#   ./serve-llama-swap.sh                 # listen on 127.0.0.1:8090
#   ./serve-llama-swap.sh 127.0.0.1:9090  # custom listen address
#   LISTEN=0.0.0.0:8090 ./serve-llama-swap.sh
#
# Send requests to the listen address; name the model in the body:
#   "model": "gemma-4-e4b"

set -euo pipefail

# Run from this script's directory so the relative paths in llama-swap.yaml
# (./models/...) resolve correctly regardless of where it's invoked from.
cd "$(dirname "$0")"

LISTEN="${1:-${LISTEN:-127.0.0.1:8090}}"
CONFIG="${CONFIG:-./llama-swap.yaml}"

if ! command -v llama-swap >/dev/null 2>&1; then
  echo "error: llama-swap not found on PATH (expected at ~/.local/bin/llama-swap)" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "error: config not found: $CONFIG" >&2
  exit 1
fi

echo "Starting llama-swap on http://$LISTEN (config: $CONFIG)"
exec llama-swap -config "$CONFIG" -listen "$LISTEN" -watch-config
