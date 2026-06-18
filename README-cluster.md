# Distributed cluster: Qwen3.5-35B-A3B over RPC

> One of two workflows in this repo — see [`README.md`](./README.md) for the
> overview and shared setup. This is the **multi-machine** path. It shares port
> `8090` with the [single-machine llama-swap workflow](./README-llama-swap.md);
> **run only one at a time** — stop llama-swap before starting the cluster, and
> vice versa.

This splits a larger hybrid MoE model, **Qwen3.5-35B-A3B** (~35B total / ~3B
active, ~22 GB at Q4), across **multiple machines** using llama.cpp's **RPC**
backend — pooling the memory of a Mac plus one or more non-Mac boxes. Unlike the
llama-swap path it is launched directly by `serve-qwen3-cluster.sh` and has **no
idle auto-unload** — it runs until you stop it (Ctrl-C).

> **Is a cluster even worth it?** At ~22 GB, if one machine has ≥32 GB free RAM,
> RPC is *slower* than running locally (activations cross the network every
> layer). The cluster only pays off when **no single box** fits the model. If one
> box almost fits, prefer single-machine MoE expert offload to CPU RAM instead:
> add `--n-cpu-moe N` (or `-ot ".ffn_.*_exps.=CPU"`) to keep it on one machine.

---

## Version pinning (critical)

llama.cpp RPC requires **every node to run the *identical* build** — a version
mismatch corrupts the protocol and crashes. The upstream prebuilt releases also
don't ship `rpc-server` / aren't built with `-DGGML_RPC=ON`. So this repo builds
llama.cpp itself, from a pinned tag (**`b9701`**), for all node platforms, via a
GitHub Action. (This is the same binary the [single-machine
workflow](./README-llama-swap.md) uses — see [Requirements in
`README.md`](./README.md#requirements).)

## 1. Build the pinned binaries (once, in CI)

Run the **`build-llamacpp-rpc`** workflow (Actions tab → Run workflow, or
`gh workflow run build-llamacpp-rpc.yml -f tag=b9701`). It builds
`llama-server` + `rpc-server` with `-DGGML_RPC=ON` for three targets and
publishes them to a Release tagged `llamacpp-rpc-b9701`:

| Platform | Backend |
|---|---|
| macOS arm64 | Metal (embedded shader lib) |
| Linux amd64 | CPU |
| Windows amd64 | CPU |

## 2. Fetch the binaries on every node

```bash
./fetch-llamacpp-rpc.sh        # auto-detects platform, extracts to ./llama.cpp/bin
```

Downloads the asset over plain HTTPS from the public Release — **no `gh`, no auth**
(it derives `owner/repo` from the git remote; override with `GH_REPO`). If the
Release is private it falls back to `gh` when installed. On **Windows**, run the
PowerShell equivalent instead:

```powershell
.\fetch-llamacpp-rpc.ps1        # pulls llama-b9701-windows-amd64-cpu.zip into .\llama.cpp\bin
```

Each artifact contains a `BUILD_COMMIT.txt` — confirm it's **identical on every
node**.

## 3. Download the model (main node only)

```bash
./download-qwen3.5-35b-a3b.sh ./models
```

The main node reads the GGUF and streams weights to the workers over RPC, so
**workers need no local model copy**. Qwen3.5 is a single hybrid model with
thinking **on by default** (the serve script uses the recommended thinking-mode
sampling); to run non-thinking, add
`--chat-template-kwargs '{"enable_thinking":false}'` to the server command.

## 4. Start the workers (each non-Mac box)

```bash
./start-rpc-worker.sh          # listens on 0.0.0.0:50052
```

On **Windows** workers, use the PowerShell script instead:

```powershell
.\start-rpc-worker.ps1         # listens on 0.0.0.0:50052
```

First run on Windows: open the port through the firewall once, from an elevated
PowerShell:

```powershell
New-NetFirewallRule -DisplayName "llama.cpp RPC worker" `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 50052
```

> **Security:** the RPC server has no authentication. Bind it only on a trusted
> LAN/VPN — never expose it to the public internet.

## 5. Run the cluster (main / Mac node)

```bash
RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 ./serve-qwen3-cluster.sh
```

Listens on `http://127.0.0.1:8090` with alias `qwen3.5-35b-a3b-cluster`. Stop it
with Ctrl-C.

### Worked example: 3 computers

| Role | Machine | RAM | LAN IP | Process |
|---|---|---|---|---|
| main | Mac Studio (arm64, Metal) | 32 GB | 192.168.1.10 | `serve-qwen3-cluster.sh` |
| worker 1 | Linux amd64 (CPU) | 32 GB | 192.168.1.20 | `rpc-server` |
| worker 2 | Windows amd64 (CPU) | 16 GB | 192.168.1.30 | `rpc-server` |

On each worker: `./start-rpc-worker.sh`. On the Mac:

```bash
RPC_WORKERS=192.168.1.20:50052,192.168.1.30:50052 \
TENSOR_SPLIT=32,32,16 \
  ./serve-qwen3-cluster.sh
```

which runs the equivalent of:

```bash
./llama.cpp/bin/llama-server \
  -m ./models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --rpc 192.168.1.20:50052,192.168.1.30:50052 \
  -ngl 999 -c 16384 --jinja \
  --tensor-split 32,32,16 \
  --alias qwen3.5-35b-a3b-cluster \
  --host 127.0.0.1 --port 8090
```

`--tensor-split` values map to `[local device, then each --rpc endpoint in
order]` — here `32,32,16` weights the layer split by each node's free RAM. Omit
`TENSOR_SPLIT` to let llama.cpp auto-distribute proportionally.

## 6. Send a request

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b-a3b-cluster",
    "messages": [{"role": "user", "content": "Explain MoE routing in one paragraph."}]
  }'
```

Same **no-cap rule** as the rest of this repo: omit `max_tokens` so output isn't
silently truncated.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| RPC connection refused / crash on load | A node is running a different llama.cpp build. Confirm `BUILD_COMMIT.txt` is **identical on every node** (re-run `./fetch-llamacpp-rpc.sh`). |
| `rpc-server: error while loading shared libraries: libgomp.so.1` (Linux) | The OpenMP runtime isn't installed. `start-rpc-worker.sh` preflights for this; install it: `sudo apt-get install -y libgomp1` (Debian/Ubuntu), `sudo dnf install -y libgomp` (Fedora/RHEL), or `sudo apk add libgomp` (Alpine). |
| `rpc-server` exits right after listing devices — Windows exit code `0xC000001D` / `-1073741795` (illegal instruction) | The binary was built with a newer CPU's instruction set than this node has (older CPUs lack AVX-512 etc.). The CI build pins `GGML_NATIVE=OFF` + `GGML_CPU_ALL_VARIANTS` for exactly this; if you hit it, re-run the **`build-llamacpp-rpc`** workflow and re-fetch the binary. |
| Worker unreachable | Check the worker's firewall and that `rpc-server` is bound on the LAN IP/port in `RPC_WORKERS`. The RPC server has no auth — keep it on a trusted LAN/VPN. |
| Download stalls partway | Re-run `./download-qwen3.5-35b-a3b.sh` — `wget --continue` resumes. |
| Crash mentioning DeltaNet / recurrent state | Qwen3.5 is a hybrid (Gated DeltaNet + MoE) model; splitting recurrent layers over RPC is a known rough edge upstream. Try without RPC on a single box, or update the pinned build. |
| Want CPU-only on the main node | Replace `-ngl 999` with `-ngl 0` (set `MODEL`/flags via the serve script's env vars). |
