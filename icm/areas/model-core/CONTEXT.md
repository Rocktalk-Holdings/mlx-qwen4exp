# model-core — change the transformer

One job: edit the live MLX modules that implement `qwen4_exp`.

## Edit surfaces

- `mlx_qwen4exp/config.py` — `ModelArgs` defaults and `from_dict`
- `mlx_qwen4exp/model.py` — layer loop, PLE attach, `sanitize`, cache make
- `mlx_qwen4exp/hyper.py` — wide residual mix / combine
- `mlx_qwen4exp/attention.py` — QSA dense/sparse + indexer
- `mlx_qwen4exp/deltanet.py` — GatedDeltaNet + sigmoid gate + readout scale
- `mlx_qwen4exp/ple.py` — `ngram_rows` hash + `PLEBlock`
- `mlx_qwen4exp/moe.py` — SwitchGLU experts + shared expert
- `mlx_qwen4exp/mtp.py` — draft head (only if the brief says MTP weights)
- `mlx_qwen4exp/__init__.py` — public `ModelArgs` / lazy `Model`

## Do not touch

- `mlx_qwen4exp/convert.py` — convert-quantize area; shares `Model.sanitize` but is a different job
- `tools/run_mlx.py` — decode-tools; load path must stay “already sanitized”
- `Model.sanitize` +1-fold / fused-split rules (`model.py:304`) unless the brief is a checkpoint-compat bug
- `ngram_rows` bit math (`ple.py:47`) unless you have a C-vs-NumPy equality plan
- Vision / `model.visual.*` — ghost; dropped on purpose

## Inputs

- Working (this run): ../../01_triage/output/brief.md
- Reference (every run): ../../_shared/conventions.md
- Reference (every run): ../../_shared/verify.md
- Reference (every run): ../../objects/_index.md (one card, not the folder)

Do NOT load: `docs/VALIDATION.md`; convert scripts; the HF card.

## Process

1. Open the object card for the noun you are changing ([../../effects/CONTEXT.md](../../effects/CONTEXT.md)).
2. Edit only the surfaces above.
3. Pair every module edit with its `tests/test_*.py`.

## Verify

```bash
python3 -m pytest tests/test_hyper.py tests/test_ple.py tests/test_deltanet.py \
  tests/test_moe.py tests/test_attention.py tests/test_model.py tests/test_mtp.py -q
```

Add the matching file if you only touched one module.

## Outputs

- Subject-tree diffs. Notes → ../../03_verify/output/ on a maintenance run.

## Human check

Phone: does the PR list which of the four unusual pieces moved (hyper / QSA / GDN / PLE / MoE / MTP)? If `sanitize` changed, the brief must say why convert and load still agree.
