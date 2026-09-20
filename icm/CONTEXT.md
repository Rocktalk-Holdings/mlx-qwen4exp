# icm — system map + maintenance pipeline

One job: tell a cold agent what this repo is, which names collide, and which shelf to open next. Product code is the source of truth; cards cite it.

## Universes

| Tag | Meaning here |
|---|---|
| **live** | Implement against this: `mlx_qwen4exp/`, `tools/`, `tests/`, `README.md`. |
| **leftover** | Still on disk, not the main path: `--mtp` (v1), historical gates in `docs/VALIDATION.md`, “lossless” wording in `docs/hf-model-card.md`. |
| **ghost** | Named in comments, not in this tree: `SPEC.md`, `llama.cpp` paths, vision / ViT, `notes/MTP-SPEC-*.md`. Do not implement against ghosts. |

## Name collisions

- **PLE** = n-gram hash embedding (`ple.py`), not “please.”
- **QSA** = sparse full-attention on 12/48 layers (`attention.py`), not GatedDeltaNet.
- **GatedDeltaNet / GDN** here uses a **sigmoid** output gate (not Qwen3-Next’s silu) plus `1/sqrt(head_v_dim)`.
- **Model** (`mlx_qwen4exp.model.Model`) ≠ `Qwen4ExpModel` (the `language_model` subtree).
- **`ple_layer_ids`** is 1-based HF; **`ple_layers`** is 0-based.
- **`Model.sanitize`** is convert-time only. `tools/run_mlx.py` must **not** sanitize already-written shards (double-folds gammas).
- **MTP-v2** (`--mtp-v2`) is the serving path. MTP-v1 (`--mtp`) is leftover and slower than greedy.
- **`--equality-test` exit 1** on mlx 0.32.x is documented kernel drift, not an unlogged bug. See README Numerics note.

## How to walk

1. Task → row in root [AGENTS.md](../AGENTS.md).
2. Open that area `CONTEXT.md` (edit surfaces / do-not-touch / verify / human gate).
3. If you need “what else moves,” open [effects/CONTEXT.md](effects/CONTEXT.md) then **one** object card.

Do not load this whole folder.

## Inputs

- Working (this run): the human’s request or `01_triage/output/brief.md` if a maintenance run is in flight.
- Reference (every run): [COLD_START.md](COLD_START.md), [_meta/schema.md](_meta/schema.md), [_shared/conventions.md](_shared/conventions.md).

Do NOT load: `docs/VALIDATION.md` unless the task is historical QA; the full `objects/` shelf.

## Process

1. Classify the task (code vs docs vs convert vs decode).
2. Open one area contract. Follow its do-not-touch list.
3. Maintenance work starts at `01_triage/` and walks numbered folders.

## Outputs

- Orientation only. Run artifacts live in `icm/0*_*/output/`.

## Human check

On a phone: this page should still answer “live vs leftover vs ghost” and “where do I go.” If a new top-level product folder appears, add a routing row to `AGENTS.md` and an area contract — do not paste a spec here.
