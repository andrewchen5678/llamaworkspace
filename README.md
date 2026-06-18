# Local llama.cpp setups (Apple Metal)

Two separate workflows live in this folder. They share one binary and one port
(`8090`) — **run only one at a time.**

| Workflow | What it does | Guide |
|---|---|---|
| **Single-machine (llama-swap)** | Google **Gemma 4 E4B** on one Mac with **MTP speculative decoding**, fronted by **llama-swap** for on-demand load + idle auto-unload. Primary use: deterministic summarization with thinking on. | [`README-llama-swap.md`](./README-llama-swap.md) |
| **Distributed cluster (RPC)** | **Qwen3.5-35B-A3B** (hybrid MoE) split across a Mac + non-Mac box(es) via llama.cpp **RPC**. No idle unload. | [`README-cluster.md`](./README-cluster.md) |

> **One at a time.** Both listen on `8090` (chosen because `8080` is commonly
> occupied). Stop one before starting the other.

---

## Requirements

**Both workflows run the same CI-built `llama-server`**: the binary produced by
the GitHub Action and fetched into `./llama.cpp/bin` by `./fetch-llamacpp-rpc.sh`.
It is pinned to tag `b9701`, built with Metal + RPC, and is newer than the MTP
merge ([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)), so
`--spec-type draft-mtp` works out of the box. Using one binary everywhere keeps
behavior identical across the single-machine and cluster setups.

| Requirement | Notes |
|---|---|
| CI-built `llama-server` in `./llama.cpp/bin` | Fetch with `./fetch-llamacpp-rpc.sh` (auto-detects platform). Bundles MTP (`--spec-type draft-mtp`) + RPC. |
| `wget` or `curl` | For the model + binary download scripts (the public Release downloads over HTTPS — no auth). |
| OpenMP runtime (`libgomp`) — **Linux nodes only** | The Linux CI binary links it; minimal installs lack it and fail with `libgomp.so.1: cannot open shared object file`. Install: `apt-get install libgomp1` (Debian/Ubuntu), `dnf install libgomp` (Fedora/RHEL), `apk add libgomp` (Alpine). macOS/Windows builds don't need it. |
| GitHub CLI (`gh`) | *Optional.* Only a fallback if the Release is private, and for triggering the build workflow from the terminal. |

```bash
./fetch-llamacpp-rpc.sh                  # fetch the pinned build into ./llama.cpp/bin
./llama.cpp/bin/llama-server --version   # confirm the build number
```

> **Custom builds:** use a Homebrew or self-compiled `llama-server` only if you need
> an optimization the CI binary doesn't carry. If you do, point the commands in the
> guides at it — and run that *same* build on every cluster node (RPC requires
> identical builds).

---

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | This overview + shared requirements |
| `README-llama-swap.md` | Single-machine Gemma + MTP + llama-swap workflow |
| `README-cluster.md` | Distributed Qwen3.5-35B-A3B RPC cluster workflow |
| `download-gemma4-e4b-mtp.sh` | Downloads target model + MTP drafter (single-machine Gemma) |
| `llama-swap.yaml` | llama-swap config: summary + generic + code models, idle auto-unload |
| `serve-llama-swap.sh` | Launches llama-swap in front of llama-server (port 8090) |
| `download-qwen3.5-35b-a3b.sh` | Downloads Qwen3.5-35B-A3B GGUF for the cluster (main node) |
| `fetch-llamacpp-rpc.sh` | Downloads the pinned (b9701) RPC-enabled llama.cpp build for this node |
| `start-rpc-worker.sh` | Starts the RPC worker (`rpc-server`) on a non-Mac node |
| `serve-qwen3-cluster.sh` | Launches Qwen3.5-35B-A3B split across the cluster (main node, port 8090) |
| `.github/workflows/build-llamacpp-rpc.yml` | CI: builds llama.cpp b9701 (+RPC) for all node platforms |
| `models/` | Downloaded GGUF files |
