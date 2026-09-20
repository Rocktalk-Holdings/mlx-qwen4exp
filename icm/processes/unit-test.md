---
type: process
universe: live
status: verified
consumes: [model-args, hyper-connection, ple-block, gated-deltanet, moe, qsa-attention, model, mtp-draft]
produces: [pytest report]
---

# unit-test

Run the checkpoint-free suite (110 tests). This is the default verify movement.

## Input → Movement → Output

Repo checkout + pytest. Toy `ModelArgs`. Pass/fail plus one expected PLE degraded warning.

## Why this shape

Real weights are 71–355 GB. Tests force graphs with `np.array` / floats, not `mx.eval` (`tests/test_model.py:8–10`).

## Steps

1. `python3 -m pytest tests/ -v` (README “Running tests”).
2. Map failures to `tests/test_<module>.py`.
3. Degenerate MTP equivalence is the loop proof — do not drop it.

## If you change this

- **Hits:** every PR’s cheap gate
- **Does not hit:** `--equality-test` on real weights (different process)

## Surfaces

| Surface | Role |
|---|---|
| Agents | run before PR |
| Humans | read the count on a phone |

## See

- Area: [../areas/tests/CONTEXT.md](../areas/tests/CONTEXT.md)
- Source: `tests/`
