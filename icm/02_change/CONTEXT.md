# 02_change — edit the subject tree

One job: apply the approved brief to product files that already live where they live.

## Inputs

- Working (this run): ../01_triage/output/brief.md
- Reference (every run): the **one** area `CONTEXT.md` named in the brief
- Reference (every run): ../_shared/conventions.md

Do NOT load: other areas’ do-not-touch lists “just in case”; the full objects shelf.

## Process

1. Confirm `brief.md` exists and names paths.
2. Edit only those edit surfaces. Do not mass-move folders.
3. Write `output/change-log.md`: files touched, files refused, leftover/ghost encountered.

## Outputs

- Subject-tree diffs (git)
- `change-log.md` → output/

## Human check

Diff + `change-log.md` on a phone. If a do-not-touch file appears, require a brief amendment before `03_verify`.
