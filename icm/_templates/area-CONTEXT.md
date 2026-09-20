# {area} — {job in five words}

One job: {the single kind of edit this area owns}.

## Edit surfaces

- `{path}` — {why an editor opens it}

## Do not touch

- `{path}` — {what breaks if you “just fix” it}

## Inputs

- Working (this run): ../../01_triage/output/brief.md
- Reference (every run): ../../_shared/conventions.md
- Reference (every run): ../../_shared/verify.md

Do NOT load: {eager wrong shelves}.

## Process

1. Read the brief and this contract.
2. Change only the edit surfaces.
3. Verify with the command below.

## Verify

```bash
{command from _shared/verify.md}
```

## Outputs

- Code/docs in the subject tree (not in this folder).
- Notes → ../../03_verify/output/ if a maintenance run is in flight.

## Human check

{Phone-readable: what to read in the PR diff. Edit the brief if the change escaped this area.}
