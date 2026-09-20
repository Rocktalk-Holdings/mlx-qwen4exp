---
type: object
cluster: architecture
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: mlx_qwen4exp/attention.py
---

# QSAAttention

Sparse full-attention on 12 of 48 layers (every 4th). Dense path is exact when `n_kv <= budget+r-1` (=2051). Above that, indexer keeps a token budget; mask is still materialized densely (README limitation).

## Why this shape

Q / GATE interleave on `q_proj` is the #1 silent-failure (`attention.py:16–19`). Partial RoPE: first `n_rot` dims only.

## Shape

- `QSAIndexer`, `QSAAttention`, `QSACache`, `IndexerCache` (`attention.py:52`, `:104`, `:193`, `:378`)
- Dense vs sparse described at `attention.py:7–14`
- Block position for indexer RoPE: `b * r` (`attention.py:27–30`)

Citations: `mlx_qwen4exp/attention.py:7`, `:16`, `:193`

## Connected to

- **owns:** KV + indexer caches on full-attention layers
- **owned-by:** [model.md](model.md) when `not is_linear`
- **joins:** MTP draft layer (also full attention)
- **looks-like-but-is-not:** [gated-deltanet.md](gated-deltanet.md)

## If you change this

- **Hits:** long-context mask; incremental decode vs prefill; `tests/test_attention.py`; MTP-v2 T=2 verify (causal, but indexer is batch-sensitive)
- **Does not hit:** GDN `ArraysCache` state; PLE hash

## Surfaces

| Surface | Role |
|---|---|
| Agents | write `attention.py` + interleave tests |
| Serve | `QSACache` offsets rolled back on MTP reject |

## See

- Source: `mlx_qwen4exp/attention.py`
- Tests: `tests/test_attention.py`
