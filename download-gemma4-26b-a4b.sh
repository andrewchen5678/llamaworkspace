#!/usr/bin/env bash
#
# download-gemma4-26b-a4b.sh
#
# Downloads Gemma 4 26B-A4B (sparse MoE, QAT GGUF) for the distributed (cluster)
# setup. NO MTP drafter: on the MoE the few active params (~4B/token) leave little
# for speculative decoding to win back (the merge PR measured "minimal
# improvement" on the MoE vs >2x on the dense 31B), so it's run without one.
#
# Run this on the MAIN node only. With llama.cpp RPC the main node reads the
# model file and streams the weights to the worker rpc-servers at load time, so
# the workers do NOT need their own copy.
#
# Usage:
#   ./download-gemma4-26b-a4b.sh [TARGET_DIR]
#
# Environment overrides:
#   MODEL_QUANT  quant file (default: UD-Q4_K_XL ~16 GB; also: UD-Q2_K_XL, UD-Q8_K_XL)
#   HF_REPO      source repo (default: unsloth/gemma-4-26B-A4B-it-qat-GGUF)

set -euo pipefail

# ---- configuration ---------------------------------------------------------
TARGET_DIR="${1:-./models}"
HF_REPO="${HF_REPO:-unsloth/gemma-4-26B-A4B-it-qat-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-UD-Q4_K_XL}"

MODEL_FILE="gemma-4-26B-A4B-it-qat-${MODEL_QUANT}.gguf"

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

  RPC_WORKERS=<WORKER1_IP>:50052,<WORKER2_IP>:50052 ./serve-gemma4-26b-a4b-cluster.sh

  # ...which runs the equivalent of:
  ./llama.cpp/bin/llama-server \\
    -m "${TARGET_DIR}/${MODEL_FILE}" \\
    --rpc <WORKER1_IP>:50052,<WORKER2_IP>:50052 \\
    -ngl 999 -fa off -c 262144 --jinja \\
    --alias gemma-4-26b-a4b-cluster \\
    --host 127.0.0.1 --port 8090

Notes:
  * Main node only — workers stream the weights over RPC; no local copy needed.
  * The full 256K KV cache is large; if a worker has little free RAM, lower CTX
    (e.g. CTX=32768) or bias --tensor-split away from it — a worker OOM during KV
    allocation shows up as "Remote RPC server crashed".
  * All nodes must run the SAME llama.cpp build (see ./fetch-llamacpp-rpc.sh).

EOF
