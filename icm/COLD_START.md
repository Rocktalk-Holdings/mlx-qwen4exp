# Stationed-agent cold start

You are in [Rocktalk-Holdings/mlx-qwen4exp](https://github.com/Rocktalk-Holdings/mlx-qwen4exp). The owner may review Markdown and PRs on a phone. Orient in **three reads**, then stop loading.

## The three reads

1. [AGENTS.md](../AGENTS.md) — where you are; which shelf for this task.
2. [icm/CONTEXT.md](CONTEXT.md) — universes and name collisions.
3. **One** area contract from the routing table (or `01_triage/CONTEXT.md` for a maintenance run).

That is enough to act. Do not crawl `mlx_qwen4exp/` or slurp `icm/objects/`.

## What this repo is

An MLX implementation of Qwen3.8-Flash-Next (`qwen4_exp`): hyper-connections, QSA, GatedDeltaNet, PLE, MoE-512, optional MTP draft. Serving driver: `tools/run_mlx.py`. Weights live **outside** the repo (~71 GB 4-bit + optional ~95 GB `ngram_table.bin`).

## What you must not do

- Do not mass-move product code into `icm/` or numbered folders.
- Do not invent a vision path (text-only; leftover/ghost).
- Do not re-run `Model.sanitize` on already-converted shards.
- Do not treat `--equality-test` FAIL on mlx 0.32.x as a silent regression — read the README Numerics note first.
- Do not load `ngram_table.bin` into RAM; it is memmap-only.

## If you are maintaining (not mapping)

Copy [_templates/run-brief.md](_templates/run-brief.md) → `01_triage/output/brief.md`. Then follow `01_triage` → `02_change` → `03_verify` → `04_pr`. A human edits each `output/` file before the next stage.

## If you are lost

[effects/CONTEXT.md](effects/CONTEXT.md) answers “I am changing X.” Product docs stay in [README.md](../README.md).
