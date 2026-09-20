# mlx-qwen4exp — agent routing

MLX port of Qwen3.8-Flash-Next (`qwen4_exp`). Product code stays put. This file routes; it does not hold specs.

## Where to go

| Task | Open next |
|---|---|
| Stationed-agent cold start | [icm/COLD_START.md](icm/COLD_START.md) |
| Walk the map / name collisions | [icm/CONTEXT.md](icm/CONTEXT.md) |
| What a change hits | [icm/effects/CONTEXT.md](icm/effects/CONTEXT.md) |
| Maintenance run (triage→PR) | [icm/01_triage/CONTEXT.md](icm/01_triage/CONTEXT.md) |
| Model / numerics | [icm/areas/model-core/CONTEXT.md](icm/areas/model-core/CONTEXT.md) |
| Decode / MTP | [icm/areas/decode-tools/CONTEXT.md](icm/areas/decode-tools/CONTEXT.md) |
| Convert / quantize | [icm/areas/convert-quantize/CONTEXT.md](icm/areas/convert-quantize/CONTEXT.md) |
| Unit tests | [icm/areas/tests/CONTEXT.md](icm/areas/tests/CONTEXT.md) |
| README / HF card / validation notes | [icm/areas/docs-release/CONTEXT.md](icm/areas/docs-release/CONTEXT.md) |
| New card or area | copy [icm/_templates/](icm/_templates/) |
| Human product docs | [README.md](README.md) |

## Do not

- Mass-move `mlx_qwen4exp/`, `tools/`, or `tests/` into numbered ICM folders.
- Slurp `icm/objects/` — follow one hop from the table above.
- Treat `SPEC.md` (not in this repo) or the HF card word “lossless” as live truth.

## Status

Idle unless files exist under `icm/0*_*/output/` (other than `.gitkeep`).
