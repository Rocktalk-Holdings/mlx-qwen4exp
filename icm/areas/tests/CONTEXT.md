# tests — pin numerics without the checkpoint

One job: edit the 110 toy-config tests that replace a 355 GB load.

## Edit surfaces

- `tests/test_hyper.py` — 4-stream mix, grouped-norm, mutation
- `tests/test_ple.py` — hash vs C, chunked==single-shot, dilations
- `tests/test_deltanet.py` — `1/sqrt(head_v_dim)`, sigmoid gate, chunked==single-shot
- `tests/test_moe.py` — routing vs dense, top-k renorm, shared expert
- `tests/test_attention.py` — dense==sparse ≤2051, Q/GATE interleave, incremental==prefill
- `tests/test_model.py` — strict-load, +1 gamma fold, 48-layer toy forward
- `tests/test_mtp.py` — draft shapes, cache, degenerate accept-all==greedy

## Do not touch

- Adding `mx.eval` to tests — harness false-positives on the token `eval`; force via `np.array(...)` (`tests/test_model.py:8–10`)
- Deleting `TestDegenerateEquivalence` — that is the speculative-loop proof
- Promoting `--equality-test` (real weights, mlx-version sensitive) into this suite

## Inputs

- Working (this run): ../../01_triage/output/brief.md
- Reference (every run): ../../_shared/verify.md
- Reference (every run): matching module docstring in `mlx_qwen4exp/`

Do NOT load: `docs/VALIDATION.md` as a test oracle (historical machine + mlx 0.32.2).

## Process

1. Keep tests checkpoint-free (tiny `ModelArgs`).
2. If you change a landmine (interleave, +1 fold, hash, scale), add a failing-first assertion.
3. Do not skip the warning about degraded PLE — it is expected without a table.

## Verify

```bash
python3 -m pytest tests/ -v
```

## Outputs

- Test diffs. Count should stay at or above 110 unless the brief removes a module.

## Human check

Phone: CI or paste shows `110 passed` (or a new total with a one-line reason). A drop with “flaky mlx” is not acceptable without a named kernel issue.
