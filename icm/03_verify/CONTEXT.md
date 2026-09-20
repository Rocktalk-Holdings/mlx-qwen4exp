# 03_verify — run the named gate

One job: execute the verify command from the brief and record what happened.

## Inputs

- Working (this run): ../01_triage/output/brief.md
- Working (this run): ../02_change/output/change-log.md
- Reference (every run): ../_shared/verify.md
- Reference (every run): the area contract’s Verify section

Do NOT load: convert of the full checkpoint; `--equality-test` as a required green on mlx 0.32.x.

## Process

1. Run the cheap pytest command unless the brief is docs-only.
2. Run any extra command the brief named (module slice, `--help` smoke).
3. Write `output/verify-notes.md`: command, exit code, unexpected warnings, mlx versions if serving.

## Outputs

- `verify-notes.md` → output/

## Human check

Read the notes on a phone. `110 passed` (or a justified new count) is the default code gate. A serving FAIL on `--equality-test` must cite the README Numerics note, not “tests are flaky.”
