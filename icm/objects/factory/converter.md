---
type: object
cluster: factory
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/convert.py
---

# convert.py

HF BF16 → MLX package: sanitized shards, `ple_constants.json`, optional `ngram_table.bin`. Restartable; progressive sanitize so the 355 GB checkpoint is never fully resident.

## Why this shape

Downloads often lack `model.safetensors.index.json`; convert scans shards (`convert.py:16–19`). Fused expert/indexer keys are buffered until a layer is complete.

## Shape

- `stream_and_sanitize_transformer` / `resave_transformer` (`convert.py:299`, `:333`)
- `build_ngram_table` (`convert.py:170`) — `--skip-table` to omit
- `--limit-shards N` smoke (`convert.py:31–37`)
- Drops visual + default mtp (`_is_visual_or_mtp`)

Citations: `mlx_qwen4exp/convert.py:3`, `:170`, `:299`

## Connected to

- **owns:** on-disk MLX key names
- **owned-by:** humans building weights
- **joins:** [model.md](../architecture/model.md) `sanitize`; [quantizers.md](quantizers.md)
- **looks-like-but-is-not:** `load_model` (must not sanitize again)

## If you change this

- **Hits:** every published shard; `ple_constants.json`; README “Building weights”
- **Does not hit:** decode loops; unit-test toy forwards (unless sanitize rules change)

## Surfaces

| Surface | Role |
|---|---|
| Humans | run on a big machine |
| Agents | edit flags/docs; do not full-convert in this workspace |

## See

- Source: `mlx_qwen4exp/convert.py`
- Area: [../../areas/convert-quantize/CONTEXT.md](../../areas/convert-quantize/CONTEXT.md)
