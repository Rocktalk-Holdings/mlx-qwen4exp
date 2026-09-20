---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/ple.py
---

# PLEBlock

N-gram hash embedding at layer 1 (0-based; HF id 2). Hash is host NumPy (`ngram_rows`); gather is memmap. Table is ~95 GB and **not** a module parameter.

## Why this shape

uint64 wrap + xor is not clean in MLX, so the hash stays NumPy (`ple.py:5–8`). Without the table the model warns once and skips the block (degraded, still coherent).

## Shape

- `ngram_rows` (`ple.py:47`) → 16 row ids per token
- `PLEBlock.__call__` (`ple.py:223`) gathers, projects, gated dot, dilated depthwise conv
- Table owned by caller: `Model._ple_table` / `MemmapTable`
- Constants: `ple_constants.json` via convert, not `state_dict`

Citations: `mlx_qwen4exp/ple.py:5`, `:47`, `:223`

## Connected to

- **owns:** per-generation `_ple_prev` / `_ple_conv` on `Model`
- **owned-by:** [model.md](model.md) when `layer.has_ple`
- **joins:** [converter.md](../factory/converter.md) (table build + constants)
- **looks-like-but-is-not:** token embedding (`embed_tokens`)

## If you change this

- **Hits:** hash bit-exact tests; chunked==single-shot; convert table layout; degraded-mode warning
- **Does not hit:** MoE routing; QSA budget

## Surfaces

| Surface | Role |
|---|---|
| Agents | write `ple.py` + `test_ple.py` |
| Serve | memmap gather only |
| Humans | optional 95 GB download |

## See

- Source: `mlx_qwen4exp/ple.py`
- Tests: `tests/test_ple.py`
