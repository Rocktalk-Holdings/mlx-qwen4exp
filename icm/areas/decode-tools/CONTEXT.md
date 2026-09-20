# decode-tools — change serving / MTP loops

One job: edit how real weights are loaded and tokens are emitted.

## Edit surfaces

- `tools/run_mlx.py` — `load_model`, `attach_ple`, `generate`, `generate_mtp`, `generate_mtp_v2`, equality/benchmark
- `tools/probe_batch_numerics.py` — T=1 vs T=2 logit probe

## Do not touch

- `mlx_qwen4exp/model.py` `sanitize` — load must **not** sanitize again (`run_mlx.py:12–15`)
- GatedDeltaNet kernel in mlx-lm — leftover/ghost; document drift, do not vendor-patch from here
- `ngram_table.bin` / weight shards — not in git; memmap only (`MemmapTable` at `run_mlx.py:74`)
- MTP-v1 as the “fix” for v2 — leftover; slower than greedy. Prefer documenting v2.

## Inputs

- Working (this run): ../../01_triage/output/brief.md
- Reference (every run): ../../_shared/verify.md
- Reference (every run): ../../objects/serving/run-mlx.md
- Reference (every run): ../README Numerics note in [README.md](../../../README.md)

Do NOT load: convert/quantize tools; `docs/hf-model-card.md` (stale “lossless”).

## Process

1. Decide: load bug, greedy quality, or MTP-v2 cache/verify.
2. If MTP-v2: read `_snapshot_cache` / `_restore_cache` (`run_mlx.py:542`) and the cache-state limitation (`run_mlx.py:659`).
3. Keep `--mtp-v2` as verified-argmax, not bit-identical-to-greedy, unless you are on mlx ≤ 0.31.x.

## Verify

```bash
python3 -m pytest tests/test_mtp.py tests/test_model.py -q
```

With weights (optional): `tools/run_mlx.py --model-dir "$QWEN_MODEL_DIR" --prompt "The capital of France is" --max-tokens 32`. Do not gate on `--equality-test` exit 0 under mlx 0.32.x.

## Outputs

- Subject-tree diffs. Notes → ../../03_verify/output/.

## Human check

Phone: PR must say whether tokens are still “verified argmax of a full forward.” If someone writes “token-identical to greedy,” reject unless they measured it on a named mlx version.
