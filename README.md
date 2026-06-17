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

Summaries should be **faithful and reproducible**, not creative — so turn off
sampling randomness with greedy decoding (`temperature: 0`) and keep the other
samplers neutral:

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
    "max_tokens": 512,
    "seed": 42
  }'
```

| Setting | Value | Why for summarization |
|---|---|---|
| `temperature` | `0` | Greedy decoding — always picks the most probable token. Same input → same summary. |
| `top_p` | `1.0` | No nucleus truncation needed; `temperature: 0` already makes decoding deterministic. |
| `top_k` | `1` | Reinforces greedy selection. |
| `repeat_penalty` | `1.0` | Neutral. Repetition penalties can distort faithful restatement of the source. |
| `max_tokens` | `512` | Caps summary length. Raise for long-document digests. |
| `seed` | `42` | Fixed seed → fully reproducible runs. |

Source text for summarization is usually long, so launch the server with a
larger context (and remember `prompt + max_tokens` must fit inside `-c`):

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

> **Thinking:** Gemma 4 has reasoning enabled by default, which can consume your
> `max_tokens` before the summary appears (see Troubleshooting). For summaries,
> raise `max_tokens` or disable thinking in the request — e.g. pass
> `"chat_template_kwargs": {"enable_thinking": false}` if your build's chat
> template supports it.

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

---

## Files in this folder

| File | Purpose |
|---|---|
| `download-gemma4-e4b-mtp.sh` | Downloads target model + MTP drafter |
| `models/` | Downloaded GGUF files |
| `README.md` | This document |
