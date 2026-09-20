# processes — verbs

One job: describe movements that actually run. Empty verb shelves are forbidden; these four are real.

| Process | When |
|---|---|
| [load-and-generate.md](load-and-generate.md) | Load shards and emit greedy tokens |
| [speculative-decode.md](speculative-decode.md) | MTP-v2 batch-verify loop |
| [convert-and-quantize.md](convert-and-quantize.md) | HF → MLX → 4-bit |
| [unit-test.md](unit-test.md) | 110 toy tests |

## Inputs

- Working (this run): the verb named by [../effects/CONTEXT.md](../effects/CONTEXT.md)
- Reference (every run): [../_meta/schema.md](../_meta/schema.md)

Do NOT load: all four cards at once.

## Process

Open one card. Follow Steps citations to source.

## Outputs

None per run. New verb = copy `../_templates/process.md` only if the movement already exists in code.

## Human check

If a card describes a path nobody runs (vision, MTP-v1 as default), mark leftover or delete it.
