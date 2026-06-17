#!/usr/bin/env bash
#
# download-gemma4-e4b-mtp.sh
#
# Downloads Gemma 4 E4B (QAT GGUF) + its MTP drafter for llama.cpp
# speculative decoding.
#
# Gemma 4 does NOT use a separate sibling draft model. It ships a tiny
# Multi-Token Prediction (MTP) "drafter" (~60 MB) that shares the target's
# KV-cache. The target verifies every drafted token, so output is identical
# to running without it — just faster. Activated with `--spec-type draft-mtp`.
#
# Requires a llama.cpp build from 2026-06-07 or later (MTP merged in PR #23398).
#
# Usage:
#   ./download-gemma4-e4b-mtp.sh [TARGET_DIR]
#
# Environment overrides:
#   MODEL_QUANT  main-model quant file (default: UD-Q4_K_XL; also: UD-Q2_K_XL)
#   HF_REPO      source repo          (default: unsloth/gemma-4-E4B-it-qat-GGUF)

set -euo pipefail

# ---- configuration ---------------------------------------------------------
TARGET_DIR="${1:-./models}"
HF_REPO="${HF_REPO:-unsloth/gemma-4-E4B-it-qat-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-UD-Q4_K_XL}"

MODEL_FILE="gemma-4-E4B-it-qat-${MODEL_QUANT}.gguf"
MTP_FILE="mtp-gemma-4-E4B-it.gguf"

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
log "Done. Run llama.cpp with MTP speculative decoding:"
cat <<EOF

  # Explicit (works on any MTP-capable build):
  llama-server \\
    -m            "${TARGET_DIR}/${MODEL_FILE}" \\
    --model-draft "${TARGET_DIR}/${MTP_FILE}" \\
    --spec-type draft-mtp \\
    --spec-draft-n-max 4 \\
    -ngl 999 -fa off \\
    --host 127.0.0.1 --port 8080

  # Or let a recent build auto-discover the drafter straight from HF:
  llama-server -hf ${HF_REPO}:${MODEL_QUANT} \\
    --spec-type draft-mtp --spec-draft-n-max 4 -ngl 999 -fa off

Notes:
  * Requires llama.cpp from 2026-06-07 or later (MTP merge, PR #23398).
  * --spec-draft-n-max  = max tokens the drafter proposes per step (try 3-6).
  * -ngl 999            = offload all layers to GPU; drop for CPU-only.
  * Add mmproj-F16.gguf via --mmproj for image/audio (multimodal) input.

EOF
