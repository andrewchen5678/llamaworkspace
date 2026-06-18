<#
.SYNOPSIS
  Windows equivalent of start-rpc-worker.sh.

  Starts the llama.cpp RPC worker on a Windows cluster node. Run this on every
  Windows WORKER box. The main node's serve-qwen3-cluster.sh connects to these
  workers and offloads part of the model to them.

  +-------------------------------- SECURITY --------------------------------+
  | The RPC server has NO authentication and is documented as insecure. Bind |
  | it only on a trusted LAN or VPN. NEVER expose it to the public internet.  |
  +--------------------------------------------------------------------------+

  First-run note: open the listen port through the Windows firewall once, from an
  elevated PowerShell, or the main node can't reach this worker:
    New-NetFirewallRule -DisplayName "llama.cpp RPC worker" `
      -Direction Inbound -Action Allow -Protocol TCP -LocalPort 50052

.EXAMPLE
  .\start-rpc-worker.ps1

.NOTES
  Parameters (all optional):
    -Port   port to listen on                  (default: 50052)
    -BindIp bind address                       (default: 0.0.0.0 — LAN-reachable)
    -Dir    dir holding rpc-server.exe         (default: .\llama.cpp\bin)
#>
[CmdletBinding()]
param(
  [int]$Port    = 50052,
  [string]$BindIp = "0.0.0.0",
  [string]$Dir   = ".\llama.cpp\bin"
)

$ErrorActionPreference = "Stop"

# Run from this script's directory so the default relative paths resolve.
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "ERROR: $msg" -ForegroundColor Red }

$rpcServer = Join-Path $Dir "rpc-server.exe"

if (-not (Test-Path $rpcServer)) {
  Write-Err "rpc-server.exe not found at $rpcServer"
  Write-Err "Fetch the pinned build first:  .\fetch-llamacpp-rpc.ps1"
  exit 1
}

Write-Step "Starting RPC worker on ${BindIp}:${Port}"
Write-Step "Binary: $rpcServer"
Write-Step "Reminder: trusted LAN/VPN only — this endpoint has no authentication."
Write-Host ""

# -c enables a local tensor cache so repeated loads of the same weights are fast.
& $rpcServer -c -H $BindIp -p $Port
