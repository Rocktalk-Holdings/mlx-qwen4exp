---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/deltanet.py
---

# GatedDeltaNet

Linear-attention on 36 of 48 layers. Reuses `mlx_lm.models.gated_delta.gated_delta_update`. This port adds **sigmoid** (not silu) output gate and **`1/sqrt(head_v_dim)`** readout scale.

## Why this shape

Three pinned deviations from Qwen3-Next (`deltanet.py:6–16`). Kernel T=1 vs T>1 paths are **not** bit-equal on mlx 0.32.x — that is the MTP-v2 equality wall, not a missing `mx.eval`.

## Shape

- Four projections: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` (`deltanet.py:102–106`)
- `delta_out_scale = 1/sqrt(head_v_dim)` (`deltanet.py:94`)
- `Qwen4ExpRMSNormGated`: rms then `sigmoid(gate)` (`deltanet.py:42–50`)
- Pass raw `a`,`b` into `gated_delta_update` (kernel applies sigmoid/softplus)

Citations: `mlx_qwen4exp/deltanet.py:6`, `:94`, `:128`

## Connected to

- **owns:** `ArraysCache` recurrent state on linear layers
- **owned-by:** [model.md](model.md) when `is_linear`
- **joins:** [run-mlx.md](../serving/run-mlx.md) snapshot/restore
- **looks-like-but-is-not:** Qwen3-Next GDN (silu, fused in_proj, no extra scale)

## If you change this

- **Hits:** 36 layers; MTP-v2 cache drift; `tests/test_deltanet.py`; README Numerics note
- **Does not hit:** QSA interleave; MoE expert split

## Surfaces

| Surface | Role |
|---|---|
| Agents | write `deltanet.py` only with a numerics plan |
| mlx-lm | supplies the Metal kernel (do not vendor here) |

## See

- Source: `mlx_qwen4exp/deltanet.py`
- Tests: `tests/test_deltanet.py`
