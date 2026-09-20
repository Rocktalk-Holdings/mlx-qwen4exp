---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/hyper.py
---

# HyperConnection

Wide residual mixer. There are **no layer norms**; the stream is `[B, T, hc=4, D]` mixed by a gated low-rank path. The final mixer (`with_inject=False`) is the output norm.

## Why this shape

Every block does mix → block → `hc_combine`. Attribute names match checkpoint suffixes so sanitize only rewrites bare `hc_norm` (not an `nn.Module`).

## Shape

- Mix/combine contract in the module header (`hyper.py:5–12`)
- `grouped_rms_norm`, `hc_init`, `hc_combine`, class `HyperConnection` (`hyper.py:51`, `:94`, `:114`, `:167`)
- Gammas arrive **already** `1+w`; this module does not fold (`hyper.py:35–37`)

Citations: `mlx_qwen4exp/hyper.py:5`, `:35`, `:167`

## Connected to

- **owns:** residual geometry for every layer
- **owned-by:** [model.md](model.md)
- **joins:** [mtp-draft.md](mtp-draft.md) (draft also uses hc mix/combine)
- **looks-like-but-is-not:** RMSNorm / DeepSeek residual (different mixer)

## If you change this

- **Hits:** all 48 layers; output norm; MTP mixer; `tests/test_hyper.py` (31)
- **Does not hit:** QSA mask math; MoE routing

## Surfaces

| Surface | Role |
|---|---|
| Agents | write `hyper.py` + `test_hyper.py` |
| Sanitize | `HyperConnection.sanitize_prefix` drops `.weight` on `hc_norm` |

## See

- Source: `mlx_qwen4exp/hyper.py`
- Tests: `tests/test_hyper.py`
