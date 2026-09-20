# effects — if you change X, open these

Catalog only. Waterfalls live on the cards. If this page and a card disagree, fix the card.

| You are changing | Open first | Also | Does not mean |
|---|---|---|---|
| `ModelArgs` / `config.py` | [../objects/architecture/model-args.md](../objects/architecture/model-args.md) | [../areas/model-core/CONTEXT.md](../areas/model-core/CONTEXT.md) | tokenizer JSON |
| Layer loop / `sanitize` | [../objects/architecture/model.md](../objects/architecture/model.md) | [../objects/factory/converter.md](../objects/factory/converter.md) | `run_mlx` generate loop |
| Hyper-connections | [../objects/architecture/hyper-connection.md](../objects/architecture/hyper-connection.md) | `tests/test_hyper.py` | QSA mask |
| QSA / indexer | [../objects/architecture/qsa-attention.md](../objects/architecture/qsa-attention.md) | [../processes/speculative-decode.md](../processes/speculative-decode.md) | GDN scale |
| GatedDeltaNet | [../objects/architecture/gated-deltanet.md](../objects/architecture/gated-deltanet.md) | README Numerics; snapshot/restore | MoE top-k |
| PLE / n-gram | [../objects/architecture/ple-block.md](../objects/architecture/ple-block.md) | [../objects/factory/converter.md](../objects/factory/converter.md) | `embed_tokens` |
| MoE / experts | [../objects/architecture/moe.md](../objects/architecture/moe.md) | [../objects/factory/quantizers.md](../objects/factory/quantizers.md) | attention cache |
| MTP module | [../objects/architecture/mtp-draft.md](../objects/architecture/mtp-draft.md) | [../areas/decode-tools/CONTEXT.md](../areas/decode-tools/CONTEXT.md) | PLE table |
| `tools/run_mlx.py` | [../objects/serving/run-mlx.md](../objects/serving/run-mlx.md) | [../areas/docs-release/CONTEXT.md](../areas/docs-release/CONTEXT.md) | `Model.sanitize` |
| Convert / quantize scripts | [../processes/convert-and-quantize.md](../processes/convert-and-quantize.md) | HF card file list | unit-test toy graphs |
| Tests only | [../areas/tests/CONTEXT.md](../areas/tests/CONTEXT.md) | matching module card | real-weight equality |
| README / HF card | [../objects/docs/readme-docs.md](../objects/docs/readme-docs.md) | `run_mlx.py` `main()` flags | kernel code |

## Outside this tree (ask the owner)

Nothing in-repo points at these; they break silently:

- Hugging Face `RockTalk/Qwen3.8-Flash-Next-MLX-4bit` (card + shards)
- Local `QWEN_MODEL_DIR` / `ngram_table.bin` symlinks
- mlx / mlx-lm versions on the serving Mac
- Ghost `SPEC.md` / llama.cpp trees cited in comments

## Inputs

- Working (this run): the path or noun in the brief
- Reference (every run): [../objects/_index.md](../objects/_index.md)

## Process

1. Match a row. Open those cards only.
2. Execute the area contract’s verify.

## Outputs

None. This file is an index.

## Human check

If a new product file is added, add a row here and an object line — do not write the waterfall on this page.
