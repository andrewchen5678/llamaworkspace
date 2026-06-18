#!/usr/bin/env bash
#
# fetch-llamacpp-rpc.sh
#
# Downloads the pinned, RPC-enabled llama.cpp binaries for THIS machine's
# platform from the project's GitHub Release and unpacks them into
# ./llama.cpp/bin. Run this on every cluster node (main + workers).
#
# Why a download instead of a local build: llama.cpp RPC requires every node to
# run the *identical* build, and the upstream prebuilt releases don't ship
# rpc-server / aren't built with -DGGML_RPC=ON. The `build-llamacpp-rpc.yml`
# GitHub Action builds one pinned tag for all platforms and publishes them as the
# `llamacpp-rpc-<tag>` Release; this script just pulls the matching zip so all
# nodes are guaranteed to match.
#
# Windows nodes: there is no bash here — use the PowerShell port instead:
#   .\fetch-llamacpp-rpc.ps1   (pulls llama-<tag>-windows-amd64-cpu.zip)
#
# Usage:
#   ./fetch-llamacpp-rpc.sh
#
# Environment overrides:
#   LLAMACPP_TAG   llama.cpp tag the Release was built from (default: b9701)
#   LLAMACPP_DIR   where to extract binaries          (default: ./llama.cpp/bin)
#   GH_REPO        owner/repo holding the Release      (default: auto from git)

set -euo pipefail

# ---- configuration ---------------------------------------------------------
LLAMACPP_TAG="${LLAMACPP_TAG:-b9701}"
LLAMACPP_DIR="${LLAMACPP_DIR:-./llama.cpp/bin}"
RELEASE_TAG="llamacpp-rpc-${LLAMACPP_TAG}"

# ---- helpers ---------------------------------------------------------------
log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# Resolve owner/repo without needing gh: honor GH_REPO, else parse the git remote.
resolve_repo() {
  if [[ -n "${GH_REPO:-}" ]]; then
    echo "$GH_REPO"
    return 0
  fi
  local url
  url="$(git remote get-url origin 2>/dev/null || true)"
  if [[ -z "$url" ]]; then
    err "Can't determine the repo. Set GH_REPO=owner/repo and re-run."
    exit 1
  fi
  url="${url#git@github.com:}"
  url="${url#https://github.com/}"
  url="${url#http://github.com/}"
  echo "${url%.git}"
}

# Download a URL to a file with curl or wget. Returns non-zero on any failure
# (HTTP error, no downloader) so the caller can fall back.
download_url() {
  local url="$1" dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  else
    return 2
  fi
}

# Map this machine to the artifact built for it by the workflow matrix.
detect_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os/$arch" in
    Darwin/arm64)          echo "macos-arm64-metal" ;;
    Linux/x86_64|Linux/amd64) echo "linux-amd64-cpu" ;;
    *)
      err "No prebuilt artifact for $os/$arch. Supported: Darwin/arm64, Linux/x86_64."
      err "On Windows, download the windows-amd64-cpu zip from the Release manually."
      exit 1
      ;;
  esac
}

# ---- main ------------------------------------------------------------------
PLATFORM="$(detect_platform)"
ZIP="llama-${LLAMACPP_TAG}-${PLATFORM}.zip"

GH_REPO="$(resolve_repo)"
ASSET_URL="https://github.com/${GH_REPO}/releases/download/${RELEASE_TAG}/${ZIP}"

log "Repo        : $GH_REPO"
log "Release     : $RELEASE_TAG"
log "Platform    : $PLATFORM"
log "Artifact    : $ZIP"
log "Destination : $LLAMACPP_DIR"
echo

mkdir -p "$LLAMACPP_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Public releases download over plain HTTPS — no gh, no auth. Fall back to gh
# only if the direct download fails (e.g. the Release is private).
log "Downloading $ZIP from $ASSET_URL"
if ! download_url "$ASSET_URL" "$TMP/$ZIP"; then
  if command -v gh >/dev/null 2>&1; then
    log "Direct download failed; retrying via gh (handles private Releases / SSO)"
    gh release download "$RELEASE_TAG" --repo "$GH_REPO" --pattern "$ZIP" --dir "$TMP" --clobber
  else
    err "Could not download $ZIP."
    err "  URL: $ASSET_URL"
    err "  If the Release is private, install gh (https://cli.github.com) or set GH_TOKEN."
    err "  Otherwise check LLAMACPP_TAG ($LLAMACPP_TAG) and that the asset exists."
    exit 1
  fi
fi

log "Extracting"
# The zip contains a top-level dir (llama-<tag>-<platform>/); flatten it into
# LLAMACPP_DIR so binaries land directly in ./llama.cpp/bin.
unzip -o -j "$TMP/$ZIP" -d "$LLAMACPP_DIR"

chmod +x "$LLAMACPP_DIR/llama-server" "$LLAMACPP_DIR/rpc-server" 2>/dev/null || true

echo
if [[ -f "$LLAMACPP_DIR/BUILD_COMMIT.txt" ]]; then
  log "Build commit (must be identical on every node):"
  sed 's/^/    /' "$LLAMACPP_DIR/BUILD_COMMIT.txt"
fi

echo
log "Done. Binaries in $LLAMACPP_DIR:"
log "  workers : ./start-rpc-worker.sh"
log "  main    : ./serve-qwen3-cluster.sh"
