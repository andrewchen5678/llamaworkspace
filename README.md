# Gemma 4 E4B + MTP Speculative Decoding (llama.cpp, Apple Metal)

Run Google's **Gemma 4 E4B** locally with **MTP (Multi-Token Prediction)
speculative decoding** for faster generation. Speculative decoding does not
change output — the main model verifies every drafted token — it just produces
those tokens faster.

- **Target model:** `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (~3.9 GB)
- **MTP drafter:** `mtp-gemma-4-E4B-it.gguf` (~57 MB)
- **Source repo:** [`unsloth/gemma-4-E4B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF)

---

## Requirements

**Every workflow in this repo — the single-machine Gemma setup *and* the cluster —
runs the same CI-built `llama-server`**: the binary produced by the GitHub Action
and fetched into `./llama.cpp/bin` by `./fetch-llamacpp-rpc.sh`. It is pinned to tag
`b9701`, built with Metal + RPC, and is newer than the MTP merge
([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)), so `--spec-type
draft-mtp` works out of the box. Using one binary everywhere keeps behavior
identical across the single-machine and cluster setups.

| Requirement | Notes |
|---|---|
| CI-built `llama-server` in `./llama.cpp/bin` | Fetch with `./fetch-llamacpp-rpc.sh` (auto-detects platform). Bundles MTP (`--spec-type draft-mtp`) + RPC. |
| `wget` or `curl` | For the model download script. |

```bash
./fetch-llamacpp-rpc.sh                  # fetch the pinned build into ./llama.cpp/bin
./llama.cpp/bin/llama-server --version   # confirm the build number
```

> **Custom builds:** use a Homebrew or self-compiled `llama-server` only if you need
> an optimization the CI binary doesn't carry. If you do, point the commands below at
> it — and run that *same* build on every cluster node (RPC requires identical builds).

---

## 1. Download the models

```bash
./download-gemma4-e4b-mtp.sh ./models
```

Downloads the target model + MTP drafter into `./models`. Re-running resumes
partial downloads and skips completed files.

**Options (environment variables):**

```bash
# Smaller / lower-quality quant:
MODEL_QUANT=UD-Q2_K_XL ./download-gemma4-e4b-mtp.sh ./models

# Different download location:
./download-gemma4-e4b-mtp.sh ~/models
```

> **Quant choice on Apple Metal:** keep `UD-Q4_K_XL` (QAT). It is the
> recommended sweet spot. **Do not** use `bf16` — Apple GPUs have no native
> bf16 support. Full `f16` (~15 GB) gives no quality gain over this QAT-4bit
> model. Bump to `Q8_0` only for quality-sensitive tasks with spare RAM.

---

## 2. Run the server (with MTP speculative decoding)

```bash
./llama.cpp/bin/llama-server \
  -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --model-draft ./models/mtp-gemma-4-E4B-it.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  -ngl 999 -fa off \
  -c 16384 \
  --host 127.0.0.1 --port 8090
```

Then open the web UI at <http://127.0.0.1:8090> or send API requests (below).

**Flag reference:**

| Flag | Meaning |
|---|---|
| `-m` | Target (main) model |
| `--model-draft` / `-md` | MTP drafter file |
| `--spec-type draft-mtp` | Enable MTP speculative decoding |
| `--spec-draft-n-max 4` | Max tokens the drafter proposes per step (tune 3–6) |
| `-ngl 999` | Offload all layers to GPU (Metal). Drop / set `0` for CPU-only |
| `-fa off` | Flash-attention off (recommended for this MTP setup) |
| `-c 16384` | Context size |

**Auto-discovery alternative** (recent builds find the drafter from the repo
root automatically — no separate download needed):

```bash
./llama.cpp/bin/llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -ngl 999 -fa off -c 16384
```

### Run without MTP (baseline)

Just omit the draft flags:

```bash
./llama.cpp/bin/llama-server -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 999 -fa off -c 16384 --host 127.0.0.1 --port 8090
```

---

## 3. Send a request

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain how a CPU cache works."}],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

The JSON response includes a `timings` block. With MTP it also reports
`draft_n` and `draft_n_accepted` — the acceptance count proves speculation is
working.

### Settings preset: summarization (deterministic)

**Summarization is the primary use case here, and Gemma 4's thinking is left
enabled** — the reasoning pass improves summary faithfulness. Summaries should
also be **reproducible**, not creative, so decoding is greedy (`temperature: 0`)
with the other samplers neutral.

These deterministic sampling defaults are **baked into `llama-swap.yaml`**
(`--temp 0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0`), so when you go through
llama-swap a request only needs the model and your text. **Omit `max_tokens`** —
with thinking on, reasoning tokens count against it, so a small cap truncates the
summary; left unset, the server generates until done:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [
      {"role": "system", "content": "Summarize the user text faithfully and concisely. Do not add information that is not in the source."},
      {"role": "user", "content": "<text to summarize>"}
    ]
  }'
```

Running `llama-server` directly (no llama-swap), or to override the defaults,
set them explicitly per request:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Summarize the user text faithfully and concisely. Do not add information that is not in the source."},
      {"role": "user", "content": "<text to summarize>"}
    ],
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "seed": 42
  }'
```

| Setting | Value | Why for summarization |
|---|---|---|
| `temperature` | `0` | Greedy decoding — always picks the most probable token. Same input → same summary. |
| `top_p` | `1.0` | No nucleus truncation needed; `temperature: 0` already makes decoding deterministic. |
| `top_k` | `1` | Reinforces greedy selection. |
| `repeat_penalty` | `1.0` | Neutral. Repetition penalties can distort faithful restatement of the source. |
| `max_tokens` | *(omit)* | Don't cap it. With thinking on, a cap counts reasoning + summary together and truncates output. Set one only as an upper bound well above the expected summary (e.g. `4096`). |
| `seed` | `42` | Fixed seed → fully reproducible runs. |

Source text for summarization is usually long, so launch the server with a
larger context (and remember the prompt plus the full generation must fit
inside `-c`):

```bash
./llama.cpp/bin/llama-server -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --model-draft ./models/mtp-gemma-4-E4B-it.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -ngl 999 -fa off -c 16384 \
  --host 127.0.0.1 --port 8090
```

> **Bonus:** `temperature: 0` also **maximizes MTP draft acceptance** — greedy
> decoding makes the drafter's guesses easy to verify, so deterministic
> summarization is among the fastest workloads here.

> **Thinking (keep it on):** Gemma 4's reasoning is enabled by default and we
> **leave it enabled** for summarization — it produces more faithful summaries.
> The trade-off: reasoning tokens share the generation budget, so **don't set a
> tight `max_tokens`** — a real summary easily runs 1.5–2k+ tokens (reasoning +
> output) and a low cap truncates it with `finish_reason: "length"`. Leave
> `max_tokens` unset (no cap) and keep the context (`-c 16384`) large enough for
> prompt + reasoning + summary. If you ever need raw speed over quality,
> disabling thinking is possible (e.g. `"chat_template_kwargs":
> {"enable_thinking": false}` if your build's chat template supports it), but
> that's not the default path here.

### Settings preset: generic use

For general-purpose chat / generation (not summarization), `llama-swap.yaml`
defines a second model, **`gemma-4-e4b-generic`**, with Gemma's recommended
balanced sampling (`--temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0`) baked in.
Same weights and MTP drafter — only the sampling defaults differ:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-generic",
    "messages": [{"role": "user", "content": "Write a haiku about caches."}]
  }'
```

> Same as summarization: **leave `max_tokens` unset.** A cap silently truncates
> the response (`finish_reason: "length"`); without one, the only ceiling is the
> context window (`-c`), and overflowing that fails loudly with an
> `exceeds the available context size` error instead of quietly cutting output.

### Settings preset: coding tasks

For code generation, `llama-swap.yaml` defines **`gemma-4-e4b-code`** with
low-temperature, focused sampling (`--temp 0.2 --top-k 40 --top-p 0.95
--min-p 0.05 --repeat-penalty 1.0`): nearly deterministic for precise output,
but not fully greedy (avoids degenerate loops on repetitive syntax). It also
bumps `--spec-draft-n-max 6` — code drafts well, so a larger draft window pays
off (this is the highest-acceptance, fastest workload; see the benchmark table)
— and runs a **128K context (`-c 131072`)** to fit large files or whole repos.

> The 128K KV cache is large. If you hit memory pressure, enable flash
> attention (`-fa on`) to shrink it, or lower `-c`.

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-code",
    "messages": [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number iteratively."}]
  }'
```

> `--repeat-penalty` stays `1.0` on purpose: code legitimately repeats tokens
> (brackets, keywords, indentation), and penalizing repetition corrupts it.
> Same no-cap rule — leave `max_tokens` unset.

Switching between `gemma-4-e4b-summary` (deterministic, the default — also
reachable as `gemma-4-e4b`), `gemma-4-e4b-generic`, and `gemma-4-e4b-code` just
by changing the `model` field makes llama-swap stop one and start the other,
since only one fits the configured footprint.

---

## Auto-unload when idle (llama-swap)

`llama-server` keeps the model resident until the process exits — it has no
built-in idle unload. To free GPU/RAM automatically, this folder ships a
[`llama-swap`](https://github.com/mostlygeek/llama-swap) setup. llama-swap is a
proxy that starts `llama-server` on demand and shuts it down after a configurable
idle `ttl`, reloading transparently on the next request.

**Install** (no Homebrew formula — drop the prebuilt binary on your `PATH`):

```bash
# Apple Silicon (arm64). Pick the latest release tag from the releases page.
curl -sL https://github.com/mostlygeek/llama-swap/releases/latest/download/llama-swap_darwin_arm64.tar.gz \
  | tar xz -C ~/.local/bin llama-swap   # ensure ~/.local/bin is on your PATH
```

**Run** (from the project root, so the relative model paths resolve):

```bash
./serve-llama-swap.sh                 # listens on 127.0.0.1:8090
./serve-llama-swap.sh 127.0.0.1:9090  # custom address
```

The config (`llama-swap.yaml`) defines two models on the same weights, with a
5-minute idle `ttl`:

| `model` name | Sampling | Use |
|---|---|---|
| `gemma-4-e4b-summary` (alias `gemma-4-e4b`) | `temp 0, top-k 1, top-p 1.0` | Deterministic summarization (default) |
| `gemma-4-e4b-generic` | `temp 1.0, top-k 64, top-p 0.95` | General chat / generation |
| `gemma-4-e4b-code` | `temp 0.2, top-k 40, top-p 0.95, draft-n 6, -c 128K` | Coding tasks |

**Send requests** to the listen address and name the model in the body — that's
how llama-swap routes (and decides which to load):

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e4b-generic","messages":[{"role":"user","content":"hi"}],"max_tokens":256}'
```

**Web UI:** don't open `http://localhost:8090/` — that bundled page probes
`/props` with `autoload=false` and shows *"Server unavailable / 404"* until a
model is running. Instead open the model's own llama.cpp UI through the
`/upstream/<model>/` route, which loads it on demand:

- <http://localhost:8090/upstream/gemma-4-e4b-summary/>
- <http://localhost:8090/upstream/gemma-4-e4b-generic/>

> **Why port 8090?** `8080` is commonly occupied by other dev servers, so this
> setup uses `8090` throughout. The cluster setup below also listens on `8090`
> (you run only one of llama-swap *or* the cluster at a time — see below).

---

## Distributed (cluster) inference: Qwen3.5-35B-A3B over RPC

Everything above runs a single model on one Mac. This section is a **separate**
setup that splits a larger hybrid MoE model, **Qwen3.5-35B-A3B** (~35B
total / ~3B active, ~22 GB at Q4), across **multiple machines** using llama.cpp's
**RPC** backend — pooling the memory of a Mac plus one or more non-Mac boxes.

> **Separate from llama-swap.** llama-swap (above) is the single-machine Gemma
> path. The cluster is launched directly by `serve-qwen3-cluster.sh` and has no
> idle auto-unload. **Both listen on `8090`, so run only one at a time** — stop
> llama-swap before starting the cluster, and vice versa.

> **Is a cluster even worth it?** At ~22 GB, if one machine has ≥32 GB free RAM,
> RPC is *slower* than running locally (activations cross the network every
> layer). The cluster only pays off when **no single box** fits the model. If one
> box almost fits, prefer single-machine MoE expert offload to CPU RAM instead:
> add `--n-cpu-moe N` (or `-ot ".ffn_.*_exps.=CPU"`) to keep it on one machine.

### Version pinning (critical)

llama.cpp RPC requires **every node to run the *identical* build** — a version
mismatch corrupts the protocol and crashes. The upstream prebuilt releases also
don't ship `rpc-server` / aren't built with `-DGGML_RPC=ON`. So this repo builds
llama.cpp itself, from a pinned tag (**`b9701`**), for all node platforms, via a
GitHub Action.

### 1. Build the pinned binaries (once, in CI)

Run the **`build-llamacpp-rpc`** workflow (Actions tab → Run workflow, or
`gh workflow run build-llamacpp-rpc.yml -f tag=b9701`). It builds
`llama-server` + `rpc-server` with `-DGGML_RPC=ON` for three targets and
publishes them to a Release tagged `llamacpp-rpc-b9701`:

| Platform | Backend |
|---|---|
| macOS arm64 | Metal (embedded shader lib) |
| Linux amd64 | CPU |
| Windows amd64 | CPU |

### 2. Fetch the binaries on every node

```bash
./fetch-llamacpp-rpc.sh        # auto-detects platform, extracts to ./llama.cpp/bin
```

Needs the GitHub CLI (`gh`) for the private Release. On **Windows**, download the
`llama-b9701-windows-amd64-cpu-rpc.zip` asset from the Release page manually and
extract it to `.\llama.cpp\bin`. Each artifact contains a `BUILD_COMMIT.txt` —
confirm it's **identical on every node**.

### 3. Download the model (main node only)

```bash
./download-qwen3.5-35b-a3b.sh ./models
```

The main node reads the GGUF and streams weights to the workers over RPC, so
**workers need no local model copy**. Qwen3.5 is a single hybrid model with
thinking **on by default** (the serve script uses the recommended thinking-mode
sampling); to run non-thinking, add
`--chat-template-kwargs '{"enable_thinking":false}'` to the server command.

### 4. Start the workers (each non-Mac box)

```bash
./start-rpc-worker.sh          # listens on 0.0.0.0:50052
```

> **Security:** the RPC server has no authentication. Bind it only on a trusted
> LAN/VPN — never expose it to the public internet.

### 5. Run the cluster (main / Mac node)

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

### Send a request

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

## 4. Benchmark MTP vs baseline

Use the **server**, not `llama-cli`, for scripted timing (see Troubleshooting).
Run each variant, then read the per-request throughput from the server log:

```bash
# Throughput + draft acceptance appear in the server's own log:
grep -E "print_timing.*eval time|draft acceptance" <server-log>
```

Example output:

```
eval time = 2567.65 ms / 200 tokens (12.84 ms per token, 77.89 tokens per second)
draft acceptance = 0.355 (116 accepted / 327 generated), mean acceptance length = 2.40
```

**Measured on this machine (Apple Metal, Q4_K_XL, temp 0):**

| Run | Speed | Draft acceptance |
|---|---|---|
| Baseline (no MTP) | ~70 t/s | — |
| MTP, prose prompt | ~78 t/s (1.1×) | ~35% |
| MTP, code prompt | ~106 t/s (~1.5×) | ~60% |

> MTP speedup is **workload-dependent**: structured / code output drafts well
> (high acceptance → bigger speedup); free-form prose accepts fewer drafts.

---

## 5. Tuning

- **`--spec-draft-n-max`**: how many tokens to draft per step. Try `3`–`6`.
  Higher helps when acceptance is high (code), hurts when it's low (prose).
- **`-c` (context)**: raise for longer conversations; costs memory.
- **`Q8_0`**: download with `MODEL_QUANT` swap (note: file is named
  `gemma-4-E4B-it-qat-...`; Q8 lives in the same repo) for marginally higher
  quality at ~8 GB.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error: invalid argument: --spec-type` | `llama-server` predates MTP. Re-fetch the CI build: `./fetch-llamacpp-rpc.sh` (or run `./llama.cpp/bin/llama-server`, not a stale one on `PATH`). |
| Download stalls partway | Re-run the script — `wget --continue` resumes. |
| `llama-cli` hangs forever printing `> ` | `llama-cli` enters interactive mode and spins on EOF even with `-no-cnv`. **Use `llama-server` for batch/scripted runs.** |
| Empty `content` in chat response | Gemma 4 has thinking enabled; reasoning may occupy the token budget. Raise `max_tokens` or disable thinking in the request. |
| `failed to measure draft model memory` warning at startup | Harmless — the drafter still loads and works. |
| Want CPU-only | Replace `-ngl 999` with `-ngl 0`. |
| llama-swap web UI at `/` shows *"Server unavailable / 404"* | The bundled UI probes `/props` with `autoload=false`, which 404s when no model is loaded. Open `http://localhost:8090/upstream/<model>/` instead (e.g. `gemma-4-e4b-generic`) — it loads the model on demand. The `/v1/*` API works regardless. |
| llama-swap: `404` routing a request | The `model` field must match a configured name (`gemma-4-e4b-summary`, `gemma-4-e4b-generic`, or alias `gemma-4-e4b`). A missing/unknown `model` 404s. |

---

## Files in this folder

| File | Purpose |
|---|---|
| `download-gemma4-e4b-mtp.sh` | Downloads target model + MTP drafter (single-machine Gemma) |
| `llama-swap.yaml` | llama-swap config: summary + generic + code models, idle auto-unload |
| `serve-llama-swap.sh` | Launches llama-swap in front of llama-server (port 8090) |
| `download-qwen3.5-35b-a3b.sh` | Downloads Qwen3.5-35B-A3B GGUF for the cluster (main node) |
| `fetch-llamacpp-rpc.sh` | Downloads the pinned (b9701) RPC-enabled llama.cpp build for this node |
| `start-rpc-worker.sh` | Starts the RPC worker (`rpc-server`) on a non-Mac node |
| `serve-qwen3-cluster.sh` | Launches Qwen3.5-35B-A3B split across the cluster (main node, port 8090) |
| `.github/workflows/build-llamacpp-rpc.yml` | CI: builds llama.cpp b9701 (+RPC) for all node platforms |
| `models/` | Downloaded GGUF files |
| `README.md` | This document |
