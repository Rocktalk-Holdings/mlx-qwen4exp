# convert-quantize — build MLX weights

One job: change HF→MLX conversion or 4-bit / MTP quantization.

## Edit surfaces

- `mlx_qwen4exp/convert.py` — shard scan, progressive `Model.sanitize`, `ple_constants.json`, `ngram_table.bin`
- `tools/quantize_stream.py` — streaming 4-bit, `quantization` block in `config.json`
- `tools/quantize_mtp.py` — `mtp-weights.safetensors`

## Do not touch

- `Model.sanitize` internals without a convert **and** a unit-test plan — double-fold / fused-split landmine
- The 95 GB table path: never fully load it; convert streams; runtime memmaps
- `tools/run_mlx.py` load path — already-sanitized shards
- Vision tensors — dropped (`_is_visual_or_mtp`)

## Inputs

- Working (this run): ../../01_triage/output/brief.md
- Reference (every run): ../../_shared/verify.md
- Reference (every run): ../../objects/factory/converter.md
- Reference (every run): ../../objects/factory/quantizers.md

Do NOT load: serving equality tests as a substitute for convert smoke.

## Process

1. Smoke with `--skip-table --limit-shards 1` before any full run.
2. Keep convert restartable (tmp + rename). Keep quantizer one-shard-resident.
3. If you change `quant_predicate` or bits, update `docs/hf-model-card.md` quantization section **and** the README.

## Verify

```bash
python3 -m mlx_qwen4exp.convert --help
python3 tools/quantize_stream.py --help
python3 tools/quantize_mtp.py --help
python3 -m pytest tests/test_model.py -q
```

Full convert/quantize is a human-gated machine job, not an agent laptop job.

## Outputs

- Script diffs in the subject tree. Weight files stay **outside** git (see `.gitignore`).

## Human check

Phone: confirm the PR does not add `.safetensors` / `ngram_table.bin`. Confirm README convert steps still match flag names.
