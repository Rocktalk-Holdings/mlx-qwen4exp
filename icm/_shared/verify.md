# Factory — how to verify

Commands only. Area contracts choose a row; they do not invent new rituals.

## Cheap (every code change)

```bash
pip install pytest mlx mlx-lm numpy transformers
python3 -m pytest tests/ -v
```

Expect 110 tests. One `RuntimeWarning` about PLE degraded mode in snapshot tests is expected when no table is attached.

## Module slice (when you touched one file)

```bash
python3 -m pytest tests/test_hyper.py tests/test_ple.py tests/test_deltanet.py \
  tests/test_moe.py tests/test_attention.py tests/test_model.py tests/test_mtp.py -v
```

Pick the file that matches the area. See [../areas/tests/CONTEXT.md](../areas/tests/CONTEXT.md).

## Serving smoke (needs local weights, Apple Silicon, lots of RAM)

```bash
python3 tools/run_mlx.py --model-dir "$QWEN_MODEL_DIR" \
  --prompt "The capital of France is" --max-tokens 64
```

Optional: `--mtp-v2`, `--benchmark`, `--equality-test`.

`--equality-test` **exits 1** on mlx 0.32.x (GatedDeltaNet T=1 vs T>1 state drift). That is documented. Do not “fix” it by weakening the test without a README update. Probe: `python3 tools/probe_batch_numerics.py "$QWEN_MODEL_DIR"`.

## Convert / quantize smoke

```bash
python3 -m mlx_qwen4exp.convert --hf-dir DIR --out-dir OUT --skip-table --limit-shards 1
```

Full convert and `tools/quantize_stream.py` need hundreds of GB disk and must not run on a laptop “to see.”

## Docs-only

No pytest required. Human gate: README commands still match `tools/run_mlx.py` flags; HF card does not re-introduce a “token-identical / lossless” claim the README walked back.

## Do not

- Run the 355 GB convert in CI.
- Load `ngram_table.bin` into RAM to “check it.”
- Gate CI on `--equality-test` exit code without an mlx-version matrix.
