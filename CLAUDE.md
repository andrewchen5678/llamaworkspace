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

Local setup to run Google's **Gemma 4 E4B** on Apple Metal via `llama.cpp`, with
**MTP speculative decoding** for speed, fronted by **llama-swap** for on-demand
load and idle auto-unload.

- `download-gemma4-e4b-mtp.sh` — downloads target model + MTP drafter into `models/`
- `llama-swap.yaml` — two models on the same weights: `gemma-4-e4b-summary`
  (deterministic, primary; alias `gemma-4-e4b`) and `gemma-4-e4b-generic`
- `serve-llama-swap.sh` — launches llama-swap (default `127.0.0.1:8080`)
- `README.md` — user-facing docs

## Conventions / gotchas

- **Primary use case is summarization with thinking ON.** Reasoning shares the
  generation budget.
- **Never set a generation cap.** No `--predict`; omit `max_tokens` in requests.
  A cap silently truncates (`finish_reason: "length"`). Uncapped, the only limit
  is `-c`, which fails loudly on overflow — that's the desired behavior.
- Web UI is per-model via `http://localhost:8080/upstream/<model>/`, not `/`.
- `llama-swap` is a manually-installed binary at `~/.local/bin/llama-swap`
  (not Homebrew).
- `models/` and `.claude/` are gitignored.
