# areas — edit-surface catalog

One job: pick the area contract that owns the files you will touch.

| Area | Owns | Open |
|---|---|---|
| model-core | `mlx_qwen4exp/*.py` except `convert.py` | [model-core/CONTEXT.md](model-core/CONTEXT.md) |
| decode-tools | `tools/run_mlx.py`, `tools/probe_batch_numerics.py` | [decode-tools/CONTEXT.md](decode-tools/CONTEXT.md) |
| convert-quantize | `mlx_qwen4exp/convert.py`, `tools/quantize_*.py` | [convert-quantize/CONTEXT.md](convert-quantize/CONTEXT.md) |
| tests | `tests/test_*.py` | [tests/CONTEXT.md](tests/CONTEXT.md) |
| docs-release | `README.md`, `docs/` | [docs-release/CONTEXT.md](docs-release/CONTEXT.md) |

New area = copy [_templates/area-CONTEXT.md](../_templates/area-CONTEXT.md). Do not split an area until two real jobs keep colliding.

## Inputs

- Working (this run): the task from root AGENTS.md or `../01_triage/output/brief.md`
- Reference (every run): this table; `../_templates/area-CONTEXT.md`

Do NOT load: every area contract at once.

## Process

1. Pick one row.
2. Open that area only.

## Outputs

None. This file is a catalog.

## Human check

Five areas, five jobs. If a sixth product folder appears, add a row — do not write the rules here.
