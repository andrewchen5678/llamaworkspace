# CLAUDE.md

## Project status

This is a **private, personal-use** project. It is not published, packaged, or
depended on by anyone else.

**No backward compatibility is required.** Feel free to rename models, change
config keys, restructure files, or break existing interfaces whenever it makes
the result cleaner. Do not add compatibility shims, aliases, deprecation paths,
or "kept for compatibility" cruft. Optimize for the current best design, not for
not breaking past usage.

## What this is

Two setups in one repo.

**A. Single-machine Gemma (primary).** Run Google's **Gemma 4 E4B** on Apple
Metal via `llama.cpp`, with **MTP speculative decoding** for speed, fronted by
**llama-swap** for on-demand load and idle auto-unload.

- `download-gemma4-e4b-mtp.sh` — downloads target model + MTP drafter into `models/`
- `llama-swap.yaml` — three models on the same weights: `gemma-4-e4b-summary`
  (deterministic, primary; alias `gemma-4-e4b`), `gemma-4-e4b-generic`, and
  `gemma-4-e4b-code` (low-temp for programming)
- `serve-llama-swap.sh` — launches llama-swap (default `127.0.0.1:8090`)

**B. Distributed cluster (Qwen3.5-35B-A3B over RPC).** Split a larger hybrid MoE
model across a Mac + non-Mac box(es) using llama.cpp **RPC**. Deliberately **separate
from llama-swap** (no idle unload); both listen on `8090`, so run only ONE at a
time. All nodes must run the **identical** llama.cpp build — pinned to tag
**`b9701`** and built for every platform via the GitHub Action.

- `.github/workflows/build-llamacpp-rpc.yml` — CI build of llama.cpp `b9701`
  (+`-DGGML_RPC=ON`) for macOS arm64 (Metal), Linux amd64 CPU, Windows amd64 CPU;
  publishes the `llamacpp-rpc-b9701` Release
- `fetch-llamacpp-rpc.sh` — downloads this node's matching pinned binary
- `download-qwen3.5-35b-a3b.sh` — downloads the model (main node only)
- `start-rpc-worker.sh` — runs `rpc-server` on a worker (non-Mac) node
- `serve-qwen3-cluster.sh` — runs `llama-server` split across the cluster (main node)

- `README.md` — overview + shared requirements (CI-built binary); links to the two
  workflow guides: `README-llama-swap.md` (single-machine Gemma) and
  `README-cluster.md` (Qwen3.5 cluster)

## Conventions / gotchas

- **Primary use case is summarization with thinking ON.** Reasoning shares the
  generation budget.
- **Never set a generation cap.** No `--predict`; omit `max_tokens` in requests.
  A cap silently truncates (`finish_reason: "length"`). Uncapped, the only limit
  is `-c`, which fails loudly on overflow — that's the desired behavior.
- Web UI is per-model via `http://localhost:8090/upstream/<model>/`, not `/`.
- Port is `8090` everywhere (8080 is commonly occupied). llama-swap and the
  cluster share it — run only one at a time.
- RPC binaries are pinned to llama.cpp tag `b9701` and built via the GitHub
  Action; every cluster node must run that same build or RPC crashes.
- `llama-swap` is a manually-installed binary at `~/.local/bin/llama-swap`
  (not Homebrew).
- `models/` and `.claude/` are gitignored.
