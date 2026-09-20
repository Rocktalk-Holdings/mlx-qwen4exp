---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/mtp.py
---

# MTPDraft

One-layer full-attention draft head. Off unless `ModelArgs.load_mtp=True`. Consumes the main model’s last **wide** residual plus the last accepted token embedding; shares `lm_head`.

## Why this shape

`mtp.*` is dropped by default `Model.sanitize`. The draft is a verb’s tool: speculative decode, not a second full model.

## Shape

- `MTPDraft` / `MTPDraftLayer` (`mtp.py:74`, `:122`)
- Steps 1–7 in the module header (`mtp.py:18–40`)
- `sanitize_mtp_weights`, `make_mtp_cache` (`mtp.py:243`, `:266`)
- Degenerate accept-all must equal greedy (`tests/test_mtp.py`)

Citations: `mlx_qwen4exp/mtp.py:18`, `:122`, `:266`

## Connected to

- **owns:** draft logits
- **owned-by:** `Model.mtp` when `load_mtp`
- **joins:** [run-mlx.md](../serving/run-mlx.md) `generate_mtp_v2`
- **looks-like-but-is-not:** main `Qwen4ExpDecoderLayer` (different input fusion)

## If you change this

- **Hits:** `--mtp` / `--mtp-v2`; `quantize_mtp.py`; `tests/test_mtp.py`
- **Does not hit:** PLE table; MoE expert count

## Surfaces

| Surface | Role |
|---|---|
| Agents | write `mtp.py` with decode-tools if the loop changes |
| Serve | optional `mtp-weights.safetensors` |

## See

- Source: `mlx_qwen4exp/mtp.py`
- Area: [../../areas/decode-tools/CONTEXT.md](../../areas/decode-tools/CONTEXT.md)
