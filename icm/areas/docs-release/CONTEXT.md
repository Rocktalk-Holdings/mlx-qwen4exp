# docs-release — claims a stranger can follow

One job: edit the files a non-developer reads before they download 71 GB.

## Edit surfaces

- `README.md` — architecture, measured tables, quickstart, numerics note
- `docs/hf-model-card.md` — Hugging Face card copy (may ship to the weight repo)
- `docs/VALIDATION.md` — historical dogfood log (leftover; append-only unless correcting a fact)

## Do not touch

- Re-introducing “token-identical to greedy” / “lossless” for MTP-v2 without a named mlx + mlx-lm version and a passing `--equality-test`
- Pasting new SHA256 weight manifests into `VALIDATION.md` from memory
- Product code “to make the docs true” from this area — file a model-core or decode-tools brief instead

## Inputs

- Working (this run): ../../01_triage/output/brief.md
- Reference (every run): ../../_shared/conventions.md
- Reference (every run): `tools/run_mlx.py` `--help` / `main()` flags (`tools/run_mlx.py:926`)

Do NOT load: the whole VALIDATION file unless you are reconciling a dated claim.

## Process

1. README is canonical for serving claims. HF card **leftover** still says “lossless” / “token sequences identical” (`docs/hf-model-card.md:22`, `:67`) — fix that wording if you touch the card.
2. Keep flag names identical to `run_mlx.py`.
3. Benchmark tables must name mlx / mlx-lm versions.

## Verify

Docs-only: read README Quickstart against `main()` flags. No pytest required unless you also changed code.

## Outputs

- Markdown diffs. Do not commit weights.

## Human check

Phone: skim the README lede and the Numerics note. If they contradict the HF card, the card needs a follow-up or this PR should update it.
