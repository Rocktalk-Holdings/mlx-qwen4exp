---
type: process
universe: live
status: verified
consumes: [model-args, model, ple-block, run-mlx]
produces: [tokens on stdout]
---

# load-and-generate

Load a converted (usually 4-bit) directory and greedily decode.

## Input → Movement → Output

A `model-dir` (or `QWEN_MODEL_DIR`) plus a prompt. `load_model` + `attach_ple` then `generate`. Streamed token ids / text.

## Why this shape

Lazy shard load thrashes; eager + `mx.eval` is the house rule. Sanitizing at load double-folds gammas.

## Steps

1. Parse `--model-dir` (`tools/run_mlx.py:950`).
2. `ModelArgs.from_dict` + construct `Model` (`run_mlx.py:110–121`).
3. If `config.quantization`, `nn.quantize` with Conv1d excluded (`run_mlx.py:128–151`).
4. `load_weights` all `model-*-of-*.safetensors` (and MTP shard if enabled). **Do not sanitize.**
5. `attach_ple` memmap + constants (`run_mlx.py:206`).
6. `generate` token-by-token (`run_mlx.py:314`).

## If you change this

- **Hits:** README quickstart; RSS prints; degraded PLE warning
- **Does not hit:** convert shard layout (unless you start sanitizing at load)

## Surfaces

| Surface | Role |
|---|---|
| Humans | run the command |
| Agents | edit `load_model` / `generate` |

## See

- Objects: [../objects/serving/run-mlx.md](../objects/serving/run-mlx.md)
- Source: `tools/run_mlx.py`
