---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/moe.py
---

# Qwen4ExpMoE

512 experts, 10 active, plus one shared expert. `split_gate_up` cuts fused `gate_up_proj` so stock `SwitchGLU` works. Top-k **is** renormalized (`norm_w=true` in the C++ graph) even if a ghost SPEC said otherwise.

## Why this shape

HF default `norm_topk_prob` and llama.cpp `build_moe_ffn(..., norm_w=true)` agree; comments call the old SPEC note wrong (`moe.py:6–13`). 4-bit SwitchLinear is the serving-quality bottleneck (README).

## Shape

- `split_gate_up` (`moe.py:38`) — gate first, up second
- `Qwen4ExpMoE` (`moe.py:66`) + `_SharedMLP` (`moe.py:117`)
- Sanitize maps `mlp.experts.*` → `switch_mlp.*`

Citations: `mlx_qwen4exp/moe.py:6`, `:38`, `:66`

## Connected to

- **owns:** router `mlp.gate`, shared expert gate
- **owned-by:** every decoder layer
- **joins:** [quantizers.md](../factory/quantizers.md) (experts 4-bit, gates 8-bit)
- **looks-like-but-is-not:** dense MLP

## If you change this

- **Hits:** 48 FFN blocks; convert fused split; quant map; `tests/test_moe.py`
- **Does not hit:** attention caches; PLE table

## Surfaces

| Surface | Role |
|---|---|
| Agents | write `moe.py` + routing tests |
| Quantize | 512 experts dominate 4-bit size/quality |

## See

- Source: `mlx_qwen4exp/moe.py`
- Tests: `tests/test_moe.py`
