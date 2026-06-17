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

| Requirement | Notes |
|---|---|
| llama.cpp build **≥ 2026-06-07** | MTP merged in [PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398). Check with `llama-server --version` (needs build ≥ b9600-ish). |
| `--spec-type draft-mtp` flag | Verify: `llama-server --help \| grep spec-type` |
| `wget` or `curl` | For the download script. |

Install / update llama.cpp on macOS:

```bash
brew install llama.cpp      # or: brew upgrade llama.cpp
llama-server --version      # confirm build date / number
```

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
llama-server \
  -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --model-draft ./models/mtp-gemma-4-E4B-it.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  -ngl 999 -fa off \
  -c 16384 \
  --host 127.0.0.1 --port 8080
```

Then open the web UI at <http://127.0.0.1:8080> or send API requests (below).

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
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -ngl 999 -fa off -c 16384
```

### Run without MTP (baseline)

Just omit the draft flags:

```bash
llama-server -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 999 -fa off -c 16384 --host 127.0.0.1 --port 8080
```

---

## 3. Send a request

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
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
curl -s http://127.0.0.1:8080/v1/chat/completions \
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
curl -s http://127.0.0.1:8080/v1/chat/completions \
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
llama-server -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --model-draft ./models/mtp-gemma-4-E4B-it.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -ngl 999 -fa off -c 16384 \
  --host 127.0.0.1 --port 8080
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
curl -s http://127.0.0.1:8080/v1/chat/completions \
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

Switching between `gemma-4-e4b-summary` (deterministic, the default — also
reachable as `gemma-4-e4b`) and `gemma-4-e4b-generic` just by changing the
`model` field makes llama-swap stop one and start the other, since only one
fits the configured footprint.

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
./serve-llama-swap.sh                 # listens on 127.0.0.1:8080
./serve-llama-swap.sh 127.0.0.1:9090  # custom address
```

The config (`llama-swap.yaml`) defines two models on the same weights, with a
5-minute idle `ttl`:

| `model` name | Sampling | Use |
|---|---|---|
| `gemma-4-e4b-summary` (alias `gemma-4-e4b`) | `temp 0, top-k 1, top-p 1.0` | Deterministic summarization (default) |
| `gemma-4-e4b-generic` | `temp 1.0, top-k 64, top-p 0.95` | General chat / generation |

**Send requests** to the listen address and name the model in the body — that's
how llama-swap routes (and decides which to load):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e4b-generic","messages":[{"role":"user","content":"hi"}],"max_tokens":256}'
```

**Web UI:** don't open `http://localhost:8080/` — that bundled page probes
`/props` with `autoload=false` and shows *"Server unavailable / 404"* until a
model is running. Instead open the model's own llama.cpp UI through the
`/upstream/<model>/` route, which loads it on demand:

- <http://localhost:8080/upstream/gemma-4-e4b-summary/>
- <http://localhost:8080/upstream/gemma-4-e4b-generic/>

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
| `error: invalid argument: --spec-type` | llama.cpp too old. `brew upgrade llama.cpp`. |
| Download stalls partway | Re-run the script — `wget --continue` resumes. |
| `llama-cli` hangs forever printing `> ` | `llama-cli` enters interactive mode and spins on EOF even with `-no-cnv`. **Use `llama-server` for batch/scripted runs.** |
| Empty `content` in chat response | Gemma 4 has thinking enabled; reasoning may occupy the token budget. Raise `max_tokens` or disable thinking in the request. |
| `failed to measure draft model memory` warning at startup | Harmless — the drafter still loads and works. |
| Want CPU-only | Replace `-ngl 999` with `-ngl 0`. |
| llama-swap web UI at `/` shows *"Server unavailable / 404"* | The bundled UI probes `/props` with `autoload=false`, which 404s when no model is loaded. Open `http://localhost:8080/upstream/<model>/` instead (e.g. `gemma-4-e4b-generic`) — it loads the model on demand. The `/v1/*` API works regardless. |
| llama-swap: `404` routing a request | The `model` field must match a configured name (`gemma-4-e4b-summary`, `gemma-4-e4b-generic`, or alias `gemma-4-e4b`). A missing/unknown `model` 404s. |

---

## Files in this folder

| File | Purpose |
|---|---|
| `download-gemma4-e4b-mtp.sh` | Downloads target model + MTP drafter |
| `llama-swap.yaml` | llama-swap config: summary + generic models, idle auto-unload |
| `serve-llama-swap.sh` | Launches llama-swap in front of llama-server |
| `models/` | Downloaded GGUF files |
| `README.md` | This document |
