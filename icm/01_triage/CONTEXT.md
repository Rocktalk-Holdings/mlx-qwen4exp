# 01_triage — name the change

One job: turn an ask into a brief a phone reviewer can approve before anyone edits product code.

## Inputs

- Working (this run): the human request (issue, chat, or PR comment)
- Reference (every run): ../COLD_START.md
- Reference (every run): ../areas/CONTEXT.md
- Reference (every run): ../effects/CONTEXT.md
- Reference (every run): ../_templates/run-brief.md
- Reference (every run): ../_shared/conventions.md

Do NOT load: product modules yet; `docs/VALIDATION.md`.

## Process

1. Classify: model-core / decode-tools / convert-quantize / tests / docs-release.
2. Copy `_templates/run-brief.md` → `output/brief.md`.
3. Fill edit surfaces, do-not-touch exceptions, verify command, human gate.
4. Stop. Do not edit `mlx_qwen4exp/` in this stage.

## Outputs

- `brief.md` → output/

## Human check

Read `output/brief.md` on a phone. If the area is wrong or a do-not-touch file is listed without a reason, edit the brief in place. `02_change` reads whatever you leave.
