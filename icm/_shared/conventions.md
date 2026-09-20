# Factory — maintenance conventions

Stable across runs. Area contracts point here; they do not restate these.

## Owner

Often a non-developer. They edit Markdown and review PRs on a phone. Write briefs and PR bodies in short sentences. Put the walk-test in the PR body, not in a side doc.

## Edit surfaces vs do-not-touch

Each area `CONTEXT.md` lists **Edit surfaces** (files you may change for that job) and **Do not touch** (files that look related but break numerics, load, or claims). If your change must enter a do-not-touch file, stop and write why in `01_triage/output/brief.md` before coding.

## Human MD / PR gates

Every stage `output/` file is the gate. The human (or a cold reviewer) must be able to:

1. Read the file on a phone.
2. Edit it in place.
3. Know the next stage will read whatever they left.

Do not hide the decision in chat. A stage without an output file has not happened.

## Product code stays put

`mlx_qwen4exp/`, `tools/`, `tests/`, `docs/`, `README.md` are the subject tree. ICM cites them. Do not relocate them into numbered folders.

## One home

Architecture claims live in source docstrings + `README.md`. Cards cite `path:line`. If README and `docs/hf-model-card.md` disagree, README wins; the card is leftover until the card is updated.

## Instantiate by copy

New area, object, process, or run brief = copy from `_templates/`. Do not start from a blank page.
