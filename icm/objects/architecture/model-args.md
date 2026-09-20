---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/config.py
---

# ModelArgs

Dataclass every module builds against (`ModelArgs` in `config.py`). Cheap to import; public via `mlx_qwen4exp`.

## Why this shape

Derived sizes (hc_dim, PLE heads, n_rot) are computed once so layers cannot drift. HF `ple_layer_ids` is 1-based; `ple_layers` subtracts one.

## Shape

- Core: `hidden_size=2560`, `num_hidden_layers=48`, `vocab_size=248320`
- `layer_types` default: full attention iff `(i+1) % full_attention_interval == 0` (`config.py:82`)
- `ple_layers` property (`config.py:116`) vs `ple_layer_ids` field (`config.py:59`)
- `from_dict` unwraps `text_config`, forces `model_type="qwen4_exp"` (`config.py:151`)
- `load_mtp=False` by default (`config.py:78`)

Citations: `mlx_qwen4exp/config.py:12`, `:82`, `:116`, `:151`

## Connected to

- **owns:** derived dims used by hyper / GDN / PLE / MoE / QSA
- **owned-by:** `Model`, convert, `load_model`
- **joins:** HF `config.json` (outside git)
- **looks-like-but-is-not:** the multimodal root config (`model_type` overwritten)

## If you change this

- **Hits:** every module constructor; convert; quantize predicates; tests’ toy configs
- **Does not hit:** tokenizer files; `ngram_table.bin` bytes

## Surfaces

| Surface | Role |
|---|---|
| Agents | read/write `config.py` |
| Humans | rarely; defaults match the published HF card |
| Runtime | `ModelArgs.from_dict(config.json)` |

## See

- Source: `mlx_qwen4exp/config.py`
- Area: [../../areas/model-core/CONTEXT.md](../../areas/model-core/CONTEXT.md)
