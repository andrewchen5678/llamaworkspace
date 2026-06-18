<#
.SYNOPSIS
  Windows equivalent of fetch-llamacpp-rpc.sh.

  Downloads the pinned, RPC-enabled llama.cpp binaries for THIS Windows box from
  the project's GitHub Release and unpacks them into .\llama.cpp\bin. Run this on
  every Windows cluster node (worker or main).

  Why a download instead of a local build: llama.cpp RPC requires every node to
  run the *identical* build, and the upstream prebuilt releases don't ship
  rpc-server / aren't built with -DGGML_RPC=ON. The build-llamacpp-rpc.yml GitHub
  Action builds one pinned tag for all platforms and publishes them as the
  llamacpp-rpc-<tag> Release; this script just pulls the matching zip so all nodes
  are guaranteed to match.

.EXAMPLE
  .\fetch-llamacpp-rpc.ps1

.NOTES
  Parameters (all optional):
    -Tag    llama.cpp tag the Release was built from   (default: b9701)
    -Dir    where to extract binaries                  (default: .\llama.cpp\bin)
    -Repo   owner/repo holding the Release             (default: auto from git)
#>
[CmdletBinding()]
param(
  [string]$Tag  = "b9701",
  [string]$Dir  = ".\llama.cpp\bin",
  [string]$Repo
)

$ErrorActionPreference = "Stop"

# Run from this script's directory so the default relative paths resolve.
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "ERROR: $msg" -ForegroundColor Red }

# Resolve owner/repo without needing gh: honor -Repo, else parse the git remote.
if (-not $Repo) {
  $url = (git remote get-url origin 2>$null)
  if (-not $url) {
    Write-Err "Can't determine the repo. Pass -Repo owner/repo and re-run."
    exit 1
  }
  $url  = $url.Trim()
  $url  = $url -replace '^git@github\.com:', ''
  $url  = $url -replace '^https?://github\.com/', ''
  $Repo = $url -replace '\.git$', ''
}

$releaseTag = "llamacpp-rpc-$Tag"
$asset      = "llama-$Tag-windows-amd64-cpu.zip"
$assetUrl   = "https://github.com/$Repo/releases/download/$releaseTag/$asset"

Write-Step "Repo        : $Repo"
Write-Step "Release     : $releaseTag"
Write-Step "Platform    : windows-amd64-cpu"
Write-Step "Artifact    : $asset"
Write-Step "Destination : $Dir"
Write-Host ""

New-Item -ItemType Directory -Force -Path $Dir | Out-Null

$tmpZip     = Join-Path $env:TEMP $asset
$tmpExtract = Join-Path $env:TEMP "llama-$Tag-windows-extract"

# Public releases download over plain HTTPS — no gh, no auth. Fall back to gh
# only if the direct download fails (e.g. the Release is private).
Write-Step "Downloading $asset"
try {
  Invoke-WebRequest -Uri $assetUrl -OutFile $tmpZip
}
catch {
  if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Step "Direct download failed; retrying via gh (handles private Releases / SSO)"
    gh release download $releaseTag --repo $Repo --pattern $asset --dir $env:TEMP --clobber
    if ($LASTEXITCODE -ne 0) { Write-Err "gh download failed."; exit 1 }
  }
  else {
    Write-Err "Could not download $asset."
    Write-Err "  URL: $assetUrl"
    Write-Err "  If the Release is private, install gh (https://cli.github.com) or set GH_TOKEN."
    Write-Err "  Otherwise check -Tag ($Tag) and that the asset exists."
    exit 1
  }
}

Write-Step "Extracting"
# The zip contains a top-level dir (llama-<tag>-windows-amd64-cpu/); flatten it
# into $Dir so binaries land directly in .\llama.cpp\bin.
if (Test-Path $tmpExtract) { Remove-Item $tmpExtract -Recurse -Force }
Expand-Archive -Path $tmpZip -DestinationPath $tmpExtract -Force
Get-ChildItem -Path $tmpExtract -Recurse -File | Move-Item -Destination $Dir -Force
Remove-Item $tmpZip, $tmpExtract -Recurse -Force

Write-Host ""
$commitFile = Join-Path $Dir "BUILD_COMMIT.txt"
if (Test-Path $commitFile) {
  Write-Step "Build commit (must be identical on every node):"
  Get-Content $commitFile | ForEach-Object { "    $_" }
}

Write-Host ""
Write-Step "Done. Binaries in $Dir"
Write-Step "  start the worker : .\start-rpc-worker.ps1"
