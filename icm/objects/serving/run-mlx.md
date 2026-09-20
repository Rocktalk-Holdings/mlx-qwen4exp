---
type: object
cluster: serving
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: tools/run_mlx.py
---

# run_mlx.py

Bring-up driver for **real** weights. Not a unit test. Loads 49 shards, attaches PLE memmap, greedy or MTP decode.

## Why this shape

House rule: eager load + `mx.eval` after `load_weights` for 300 GB-class checkpoints (`run_mlx.py:3–6`). Convert already sanitized; load must not sanitize again (`run_mlx.py:12–15`).

## Shape

- `load_model` (`run_mlx.py:110`) — `from_dict`, optional `nn.quantize`, strict load
- `MemmapTable` (`run_mlx.py:74`) — gather-only
- `generate` / `generate_mtp` (leftover v1) / `generate_mtp_v2` (`run_mlx.py:314`, `:357`, `:628`)
- `_snapshot_cache` deep-copy (`run_mlx.py:542`) — mlx 0.32.x buffer reuse
- CLI `main` (`run_mlx.py:926`)

Citations: `tools/run_mlx.py:12`, `:110`, `:542`, `:628`, `:926`

## Connected to

- **owns:** serving flags strangers copy from the README
- **owned-by:** humans running generation
- **joins:** [model.md](../architecture/model.md), [mtp-draft.md](../architecture/mtp-draft.md)
- **looks-like-but-is-not:** `tests/` (toy config, no checkpoint)

## If you change this

- **Hits:** README quickstart flags; equality/benchmark; HF card usage
- **Does not hit:** `Model.sanitize` fold rules (unless you wrongly call it)

## Surfaces

| Surface | Role |
|---|---|
| Humans | copy-paste commands |
| Agents | write loops + snapshot/restore |
| CI | should not run this without weights |

## See

- Source: `tools/run_mlx.py`
- Area: [../../areas/decode-tools/CONTEXT.md](../../areas/decode-tools/CONTEXT.md)
