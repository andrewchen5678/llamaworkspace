#!/usr/bin/env bash
#
# download-qwen3-30b-a3b.sh
#
# Downloads Qwen3-30B-A3B (MoE GGUF) for the distributed (cluster) setup.
#
# Run this on the MAIN node only. With llama.cpp RPC the main node reads the
# model file and streams the weights to the worker rpc-servers at load time, so
# the workers do NOT need their own copy.
#
# Qwen3-30B-A3B is a Mixture-of-Experts model: ~30B total parameters, ~3B active
# per token. There is no separate draft model / MTP drafter — it's a single GGUF.
#
# Usage:
#   ./download-qwen3-30b-a3b.sh [TARGET_DIR]
#
# Environment overrides:
#   MODEL_QUANT  quant file (default: UD-Q4_K_XL ~18 GB; also: UD-Q8_K_XL, etc.)
#   HF_REPO      source repo (default: unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)
#                For thinking-on, use unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF.

set -euo pipefail

# ---- configuration ---------------------------------------------------------
TARGET_DIR="${1:-./models}"
HF_REPO="${HF_REPO:-unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-UD-Q4_K_XL}"

MODEL_FILE="Qwen3-30B-A3B-Instruct-2507-${MODEL_QUANT}.gguf"

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# ---- helpers ---------------------------------------------------------------
log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# Download one file. Prefers the huggingface CLI (resumable), falls back to
# wget/curl against the resolve/ endpoint.
download() {
  local repo="$1" file="$2" dest_dir="$3"
  local dest="${dest_dir}/${file}"

  if [[ -f "$dest" ]]; then
    log "Already present, skipping: $dest"
    return 0
  fi

  mkdir -p "$dest_dir"

  if command -v hf >/dev/null 2>&1; then
    log "Downloading $repo / $file via hf"
    hf download "$repo" "$file" --local-dir "$dest_dir"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    log "Downloading $repo / $file via huggingface-cli"
    huggingface-cli download "$repo" "$file" --local-dir "$dest_dir"
  else
    local url="${HF_ENDPOINT}/${repo}/resolve/main/${file}"
    log "HF CLI not found; downloading from $url"
    if command -v wget >/dev/null 2>&1; then
      wget --continue --show-progress -O "$dest" "$url"
    elif command -v curl >/dev/null 2>&1; then
      curl -L -C - -o "$dest" "$url"
    else
      err "Need one of: hf, huggingface-cli, wget, or curl."
      exit 1
    fi
  fi
}

# ---- main ------------------------------------------------------------------
log "Repo        : $HF_REPO"
log "Model       : $MODEL_FILE"
log "Destination : $TARGET_DIR"
echo

download "$HF_REPO" "$MODEL_FILE" "$TARGET_DIR"

echo
log "Done. Run the cluster from the main node (workers via ./start-rpc-worker.sh):"
cat <<EOF

  RPC_WORKERS=<WORKER1_IP>:50052,<WORKER2_IP>:50052 ./serve-qwen3-cluster.sh

  # ...which runs the equivalent of:
  ./llama.cpp/bin/llama-server \
    -m "${TARGET_DIR}/${MODEL_FILE}" \
    --rpc <WORKER1_IP>:50052,<WORKER2_IP>:50052 \
    -ngl 999 -c 16384 \
    --alias qwen3-30b-a3b-cluster \
    --host 127.0.0.1 --port 8090

Notes:
  * Main node only — workers stream the weights over RPC; no local copy needed.
  * --rpc takes one comma-separated endpoint per worker.
  * Add --tensor-split A,B,C (maps to [local, worker1, worker2]) to bias the
    layer split by each node's free RAM. Omit it to auto-distribute.
  * All nodes must run the SAME llama.cpp build (see ./fetch-llamacpp-rpc.sh).

EOF
