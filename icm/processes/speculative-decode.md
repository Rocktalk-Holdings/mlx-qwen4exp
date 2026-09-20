---
type: process
universe: live
status: verified
consumes: [mtp-draft, gated-deltanet, qsa-attention, run-mlx]
produces: [verified-argmax tokens]
---

# speculative-decode

MTP-v2: draft one token, batch-verify T=2, accept or rollback. Leftover sibling: `--mtp` (v1 sequential), slower than greedy.

## Input → Movement → Output

Prefill like greedy. Loop: draft from `_last_hidden_wide`, snapshot caches, `model([[cur, draft]])`, accept two tokens or restore and verify one. Stream plus accept-rate stats.

## Why this shape

T=2 is the speedup. It is **not** bit-equal to two T=1 GDN steps on mlx 0.32.x (`generate_mtp_v2` cache-state limitation, `run_mlx.py:659`). Snapshot must deep-copy (`run_mlx.py:542`).

## Steps

1. Prefill → first token (`run_mlx.py:628` protocol).
2. `MTPDraft` → `draft_tok`.
3. `_snapshot_cache` (ArraysCache `a+0`, QSA offsets, PLE, last wide).
4. Batch verify; on reject `_restore_cache` + T=1 verify.
5. Optional `--equality-test` (`run_mlx.py:805`) — expect FAIL on mlx 0.32.x.

## If you change this

- **Hits:** tok/s, accept %, README Numerics, `tests/test_mtp.py` degenerate loop
- **Does not hit:** convert; PLE hash constants

## Surfaces

| Surface | Role |
|---|---|
| Humans | `--mtp-v2` |
| Agents | edit snapshot/verify with a named mlx version |

## See

- Objects: [../objects/architecture/mtp-draft.md](../objects/architecture/mtp-draft.md), [../objects/architecture/gated-deltanet.md](../objects/architecture/gated-deltanet.md)
- Source: `tools/run_mlx.py:628`
