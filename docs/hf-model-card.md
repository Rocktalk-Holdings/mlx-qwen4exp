---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
tags:
  - mlx
  - qwen4_exp
  - text-generation
  - apple-silicon
  - speculative-decoding
  - quantized
language:
  - en
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next MLX 4-bit

First open-code MLX port of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`qwen4_exp` architecture). Runs at **25–35 tok/s** on Apple Silicon with MTP speculative decoding
(batch-verify mode, lossless).

**Code repository:** [Rocktalk-Holdings/mlx-qwen4exp](https://github.com/Rocktalk-Holdings/mlx-qwen4exp)

---

## What is this model?

Qwen3.8-Flash-Next is Qwen's experimental preview of the Qwen4 architecture.
~180B total parameters: 125B MoE transformer (6B active, 512 experts, top-10+shared)
+ 51.2B n-gram hash embedding table + 4B MTP draft head.

Architecture highlights:

- **Hyper-connections**: 4-stream widened residual replaces all layer norms with gated low-rank mixers
- **QSA sparse attention** on 12/48 layers: lightning indexer selects top-2048-token budget
- **Gated DeltaNet** on 36/48 layers: sigmoid-gated linear attention with `1/sqrt(head_v_dim)` readout scale
- **PLE n-gram embedding**: 320M-row hash table, gathered per-token off memmap (~95 GB, optional)
- **MTP draft head**: 1-layer decoder for speculative decoding (66–86% acceptance rate)

See [mlx-lm issue #1798](https://github.com/ml-explore/mlx-lm/issues/1798) for the QSA sparse mask.

---

## Quantization

4-bit affine quantization (group size 64) via [mlx_qwen4exp](https://github.com/Rocktalk-Holdings/mlx-qwen4exp).

Per-path overrides that match the reference GGUF (Q8_0 for these):
- `embed_tokens`, `lm_head` → 8-bit g64
- `mlp.gate` (router), `shared_expert_gate` → 8-bit g64
- All 512 switch expert weights → 4-bit g64
- All other linear layers → 4-bit g64

MTP weights (`mtp-weights.safetensors`) are quantized with the same policy.

---

## Performance (M3 Ultra, 512 GB, Metal)

| Mode | tok/s |
|------|-------|
| Plain greedy | 26.4–26.6 |
| MTP-v2 speculative | 27.6–35.3 (avg 30.3) |

MTP-v2 acceptance: 66–86%, content-dependent. Token sequences identical to plain greedy (verified).

---

## Usage

Install the code package:

```bash
git clone https://github.com/Rocktalk-Holdings/mlx-qwen4exp
cd mlx_qwen4exp
pip install mlx mlx-lm numpy transformers
```

Download this model:

```bash
huggingface-cli download RockTalk/Qwen3.8-Flash-Next-MLX-4bit \
    --local-dir /path/to/model-dir
```

Run:

```bash
python3 tools/run_mlx.py --model-dir /path/to/model-dir \
    --prompt "The capital of France is" --max-tokens 64

# With MTP-v2 speculative decoding (recommended):
python3 tools/run_mlx.py --model-dir /path/to/model-dir \
    --prompt "The capital of France is" --mtp-v2 --max-tokens 64
```

---

## Files in this repository

| File | Size | Description |
|------|------|-------------|
| `model-00001-of-00049.safetensors` … `model-00049-of-00049.safetensors` | ~71 GB total | 4-bit quantized transformer weights |
| `mtp-weights.safetensors` | ~1.4 GB | Quantized MTP draft head weights |
| `config.json` | — | Model config with quantization spec |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | — | Tokenizer |
| `ple_constants.json` | — | PLE n-gram hash constants (required for full quality) |
| `ngram_table.bin` | ~95 GB | Float16 n-gram embedding table (optional; model runs without it in degraded mode) |

The n-gram table is a large optional download. Without it, the PLE block is skipped and
a single warning is printed; output quality is slightly reduced but the model remains coherent.

---

## Limitations

- Text-only: vision path not implemented
- QSA sparse mask is O(n²) above 2051 tokens (correctness preserved, memory scales with full KV)
- 4-bit expert quantization is the only serving-viable MLX option (6/8-bit is correct but ~50× slower)

---

## License

Model weights: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
Code: MIT ([Rocktalk-Holdings/mlx-qwen4exp](https://github.com/Rocktalk-Holdings/mlx-qwen4exp))
