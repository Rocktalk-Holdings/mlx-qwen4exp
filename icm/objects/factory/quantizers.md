---
type: object
cluster: factory
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: tools/quantize_stream.py
---

# Quantizers

`quantize_stream.py` (transformer 4-bit g64) and `quantize_mtp.py` (draft shard). Policy: embed/lm_head/routers 8-bit; experts and most linears 4-bit; `ngram_table.bin` not quantized (symlink).

## Why this shape

A bf16 eager load (~234 GB) may not fit beside other servers; stream one ~5 GB shard (`quantize_stream.py:3–6`). Output names match `QuantizedLinear` so `nn.quantize` at load reconstructs the tree.

## Shape

- `build_quant_map` / `quantize_shard` (`quantize_stream.py:81`, `:108`)
- `config["quantization"]` written for mlx_lm (`quantize_stream.py:18–19`)
- MTP: collect `mtp.*` from BF16, sanitize, write `mtp-weights.safetensors`

Citations: `tools/quantize_stream.py:3`, `:81`; `tools/quantize_mtp.py:71`

## Connected to

- **owns:** 4-bit serving pin
- **owned-by:** weight publishers
- **joins:** [converter.md](converter.md), [run-mlx.md](../serving/run-mlx.md) `nn.quantize`
- **looks-like-but-is-not:** mlx-lm `quantize_model` (same convention, this repo’s predicate)

## If you change this

- **Hits:** HF card quantization section; load `class_predicate`; quality/speed
- **Does not hit:** PLE hash; unit-test toy weights

## Surfaces

| Surface | Role |
|---|---|
| Humans | run `--src-dir` / `--dst-dir` |
| Agents | edit policy + docs together |

## See

- Source: `tools/quantize_stream.py`, `tools/quantize_mtp.py`
- Area: [../../areas/convert-quantize/CONTEXT.md](../../areas/convert-quantize/CONTEXT.md)
