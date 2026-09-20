# Schema — closed types

Underscore folder: about the workspace, not the product.

## Node types

| `type` | Lives in | Instantiated from |
|---|---|---|
| `object` | `objects/<cluster>/` | `_templates/object.md` |
| `process` | `processes/` | `_templates/process.md` |
| `area` | `areas/<kebab>/CONTEXT.md` | `_templates/area-CONTEXT.md` |
| `stage` | `0N_<kebab>/CONTEXT.md` | `_templates/stage-CONTEXT.md` |

Do not invent a fifth type. A new noun is an object card. A new movement is a process card.

## Frontmatter (objects / processes)

```yaml
type: object | process
cluster: architecture | serving | factory | docs   # objects only
universe: live | leftover | ghost
status: stub | verified | stale
entity: path to owning source
consumes: []   # processes
produces: []   # processes
```

`verified` requires a date, a commit or branch, and at least one `{path}:{line}` citation. No citation → `stub`.

## Universes

Defined once in [../CONTEXT.md](../CONTEXT.md). Cards inherit; they do not redefine.

## Naming

- Files: kebab-case.
- Stage folders: `NN_kebab-name`.
- One home per fact; link instead of copy.

## Status surface

A maintenance run is **in flight** if any `icm/0*_*/output/` file other than `.gitkeep` exists. Empty output folders = idle.
