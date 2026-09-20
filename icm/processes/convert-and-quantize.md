---
type: process
universe: live
status: verified
consumes: [model, converter, quantizers]
produces: [MLX bf16 dir, 4-bit dir]
---

# convert-and-quantize

Turn Qwen’s BF16 tree into the published MLX 4-bit directory.

## Input → Movement → Output

`--hf-dir` shards. Convert sanitizes and optionally concatenates the n-gram table. Quantize streams to 4-bit; optional MTP inject.

## Why this shape

Full eager convert does not fit. Restartable tmp+rename. Table build is optional (`--skip-table`).

## Steps

1. `python3 -m mlx_qwen4exp.convert --hf-dir … --out-dir …` (`convert.py:398`).
2. Smoke: `--limit-shards 1 --skip-table`.
3. `tools/quantize_stream.py --src-dir … --dst-dir …`.
4. Optional `tools/quantize_mtp.py`.
5. Serve with `load-and-generate` — no second sanitize.

## If you change this

- **Hits:** published weights; HF card file list; disk/RAM gates
- **Does not hit:** unit tests (toy weights) unless sanitize rules change

## Surfaces

| Surface | Role |
|---|---|
| Humans | big-machine job |
| Agents | flag/docs only unless asked to run a limited smoke |

## See

- Objects: [../objects/factory/converter.md](../objects/factory/converter.md)
- Source: `mlx_qwen4exp/convert.py`
