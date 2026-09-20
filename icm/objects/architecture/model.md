---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/model.py
---

# Model

Integrator `mlx_qwen4exp.model.Model`. Wires hyper / PLE / GDN / QSA / MoE. `Qwen4ExpModel` is only the `language_model` subtree.

## Why this shape

Checkpoint keys map with minimal renaming: `model.language_model.layers.N.*` plus root `lm_head`. PLE table is **not** a parameter; attach after load or run degraded.

## Shape

- Tree: `Model.model.language_model.{embed_tokens,layers,hyper_connection_mixer}` (`model.py:8–16`)
- Forward: `hc_init` → per layer (optional PLE, attn, MoE) → stash `_last_hidden_wide` → terminal mixer → `lm_head` (`model.py:188`)
- `sanitize` (`model.py:304`): drop visual/mtp/PLE junk, +1 gamma fold, split fused experts and indexer, conv layout
- PLE helpers: `_ple_table`, `reset_ple_state` (`model.py:120`, `:144`)

Citations: `mlx_qwen4exp/model.py:8`, `:105`, `:188`, `:304`

## Connected to

- **owns:** `Qwen4ExpDecoderLayer`, sanitize contract
- **owned-by:** `tools/run_mlx.py` `load_model`; `convert.py`
- **joins:** [mtp-draft.md](mtp-draft.md) via `_last_hidden_wide`
- **looks-like-but-is-not:** `Qwen4ExpModel` (inner), mlx-lm generic `Model`

## If you change this

- **Hits:** convert (progressive sanitize), load (must not sanitize twice), all tests in `test_model.py`
- **Does not hit:** GDN Metal kernels (mlx-lm); HF tokenizer

## Surfaces

| Surface | Role |
|---|---|
| Agents | write layer loop / sanitize with a brief |
| Convert | calls `Model.sanitize` per shard |
| Serve | `load_weights(strict=True)` on post-sanitize keys |

## See

- Source: `mlx_qwen4exp/model.py`
- Area: [../../areas/model-core/CONTEXT.md](../../areas/model-core/CONTEXT.md)
