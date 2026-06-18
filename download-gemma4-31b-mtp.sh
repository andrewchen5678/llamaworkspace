#!/usr/bin/env bash
#
# download-gemma4-31b-mtp.sh
#
# Downloads Gemma 4 31B (dense, QAT GGUF) + its MTP drafter for the distributed
# (cluster) setup with llama.cpp speculative decoding.
#
# Run this on the MAIN node only. With llama.cpp RPC the main node reads the
# model file and streams the weights to the worker rpc-servers at load time, so
# the workers do NOT need their own copy. The MTP drafter is tiny and runs on the
# main node alongside the server.
#
# Gemma 4 does NOT use a separate sibling draft model. It ships a tiny
# Multi-Token Prediction (MTP) "drafter" that shares the target's KV-cache. The
# target verifies every drafted token, so output is identical to running without
# it — just faster. Activated with `--spec-type draft-mtp`. The 31B is DENSE, so
# MTP pays off (>2x in the merge PR's tests) — unlike the 26B-A4B MoE, where the
# few active params leave little for the drafter to win back.
#
# Requires a llama.cpp build from 2026-06-07 or later (MTP merged in PR #23398).
# The pinned cluster build (tag b9701, 2026-06-18) is newer than that, so MTP
# works out of the box — see ./fetch-llamacpp-rpc.sh.
#
# Usage:
#   ./download-gemma4-31b-mtp.sh [TARGET_DIR]
#
# Environment overrides:
#   MODEL_QUANT  main-model quant file (default: UD-Q4_K_XL ~19 GB; also: UD-Q2_K_XL)
#   HF_REPO      source repo          (default: unsloth/gemma-4-31B-it-qat-GGUF)

set -euo pipefail

# ---- configuration ---------------------------------------------------------
TARGET_DIR="${1:-./models}"
HF_REPO="${HF_REPO:-unsloth/gemma-4-31B-it-qat-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-UD-Q4_K_XL}"

MODEL_FILE="gemma-4-31B-it-qat-${MODEL_QUANT}.gguf"
MTP_FILE="mtp-gemma-4-31B-it.gguf"

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
log "Main model  : $MODEL_FILE"
log "MTP drafter : $MTP_FILE"
log "Destination : $TARGET_DIR"
echo

download "$HF_REPO" "$MODEL_FILE" "$TARGET_DIR"
download "$HF_REPO" "$MTP_FILE"   "$TARGET_DIR"

echo
log "Done. Run the cluster from the main node (workers via ./start-rpc-worker.sh):"
cat <<EOF

  RPC_WORKERS=<WORKER1_IP>:50052,<WORKER2_IP>:50052 ./serve-gemma4-31b-cluster.sh

  # ...which runs the equivalent of:
  ./llama.cpp/bin/llama-server \\
    -m            "${TARGET_DIR}/${MODEL_FILE}" \\
    --model-draft "${TARGET_DIR}/${MTP_FILE}" \\
    --spec-type draft-mtp --spec-draft-n-max 4 \\
    --rpc <WORKER1_IP>:50052,<WORKER2_IP>:50052 \\
    -ngl 999 -fa off -c 262144 --jinja \\
    --alias gemma-4-31b-cluster \\
    --host 127.0.0.1 --port 8090

Notes:
  * Main node only — workers stream the weights over RPC; no local copy needed.
  * The MTP drafter runs on the main node; only the target model is split.
  * MTP + RPC together is not yet validated upstream — if the cluster crashes or
    stalls at load, retry with NO_MTP=1 to drop speculative decoding (see the
    serve script) and confirm the split itself works first.
  * All nodes must run the SAME llama.cpp build (see ./fetch-llamacpp-rpc.sh).

EOF
