---
type: object
cluster: docs
universe: live
status: verified
verified_date: 2026-09-20
verified_on: main
entity: README.md
---

# README + release docs

Human entry. README is canonical for serving claims. `docs/hf-model-card.md` ships with the weight repo and is **leftover** on “lossless.” `docs/VALIDATION.md` is a dated dogfood log.

## Why this shape

Strangers download 71 GB from these pages. Wrong “token-identical” language caused a real release gate (`docs/VALIDATION.md` passes 1–4).

## Shape

- README: architecture, mlx-versioned tables, quickstart, numerics, convert, 110 tests
- HF card still says “lossless” / “Token sequences identical to plain greedy” (`docs/hf-model-card.md:22`, `:67`)
- VALIDATION: leftover measurements on a private M3 Ultra

Citations: `README.md:1`, `:148`; `docs/hf-model-card.md:22`

## Connected to

- **owns:** commands humans paste
- **owned-by:** maintainers
- **joins:** [run-mlx.md](../serving/run-mlx.md) flags
- **looks-like-but-is-not:** ghost `SPEC.md`

## If you change this

- **Hits:** first-run UX; HF model page if the card is copied out
- **Does not hit:** numerics in `deltanet.py` unless you are “fixing” docs by changing kernels

## Surfaces

| Surface | Role |
|---|---|
| Humans | read on phone / HF |
| Agents | keep flags and claims aligned |

## See

- Source: `README.md`
- Area: [../../areas/docs-release/CONTEXT.md](../../areas/docs-release/CONTEXT.md)
