# 04_pr — open the pull request

One job: draft a PR a phone reviewer can accept, including a walk-test.

## Inputs

- Working (this run): ../01_triage/output/brief.md
- Working (this run): ../02_change/output/change-log.md
- Working (this run): ../03_verify/output/verify-notes.md
- Reference (every run): ../_shared/conventions.md
- Reference (every run): root AGENTS.md (confirm still ≤ ~60 lines if you edited it)

Do NOT load: weight manifests; VALIDATION SHA tables.

## Process

1. Draft `output/pr-draft.md`: title, why, files, verify, **walk-test** (three reads + one area + one effect row).
2. Open/update the PR against the default branch (`main`).
3. Leave stage output in `output/` if you want status visible; delete run files (keep `.gitkeep`) when the run is done.

## Outputs

- `pr-draft.md` → output/
- GitHub PR

## Human check

On a phone: title makes sense; walk-test names real paths; no `.safetensors` in the diff; AGENTS.md is still a catalog. Approve or comment on the PR — do not re-triage in chat.
