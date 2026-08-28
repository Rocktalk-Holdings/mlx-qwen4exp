# mlx_qwen4exp — Qwen3.8-Flash-Next MLX Port

First open-code MLX implementation of Qwen3.8-Flash-Next (`qwen4_exp` architecture).
Runs at **25–35 tok/s** on Apple Silicon with 4-bit quantization; supports
**MTP speculative decoding** (batch-verify mode: every emitted token is the verified argmax of a full forward pass — self-consistent greedy). See the Numerics note below regarding token-exact equivalence to plain greedy.

Model weights: [RockTalk/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/RockTalk/Qwen3.8-Flash-Next-MLX-4bit)
Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

---

## Architecture — what makes this model unusual

Qwen3.8-Flash-Next is an experimental preview of the Qwen4 architecture. Four features
distinguish it from standard transformers:

**Hyper-connections** — the residual stream is 4 parallel copies
(`[batch, seq, 4, hidden]`). Every layer norm is replaced by a gated low-rank mixer
(down 10240→320, up 320→10240). The final output norm is the last mixer. This is
DeepSeek-V4-style widened residuals with a Qwen low-rank twist.

**QSA sparse attention** on 12 of 48 layers (every 4th): a 4-head "lightning indexer"
scores 4-token blocks, selects a top-2048-token budget, then runs dense attention over
that budget only. The sparse mask is equivalent to dense up to 2051 tokens; beyond that
it selects a subset. See [mlx-lm issue #1798](https://github.com/ml-explore/mlx-lm/issues/1798)
for the mask behavior.

**Gated DeltaNet** on the other 36 of 48 layers: state-space linear attention with a
sigmoid output gate (not silu — one concrete difference from Qwen3-Next's GDN). The
delta-rule kernel is reused from `mlx_lm.models.gated_delta`; this port adds the
`1/sqrt(head_v_dim)` readout scale that llama.cpp's fused kernel applies and mlx-lm's
standalone kernel omits.

**PLE n-gram embedding** at layer 1: a 51.2B-parameter hash embedding table
(320M rows × 160 dims, float16, ~95 GB). Hash of (current, prev, prev-2) tokens
→ 16 table rows → gated into the wide residual via a dilated depthwise conv. The table
lives on disk and is gathered via memmap — it is never fully loaded into RAM.

**MoE-512**: 512 experts, 10 active per token, plus 1 shared expert. Each expert is
640-dimensional. Total ~125B MoE transformer parameters + 51.2B table = ~180B total.

**MTP draft head**: 1-layer full-attention decoder that drafts the next token from the
main model's wide residual + the embedding of the last accepted token. Enables
speculative decoding with 66–86% acceptance rate (content-dependent).

---

## Measured performance (M3 Ultra, 512 GB, 4-bit quantized, Metal)

| Mode | tok/s | Notes |
|------|-------|-------|
| Plain greedy | 26.4–26.6 | Prefill ~100 tok/s for short prompts |
| MTP-v2 speculative | 27.6–35.3 (avg 30.3) | +15% average; up to +34% on structured text |

*Table above measured on mlx 0.30.6 + mlx-lm 0.31.1. On mlx 0.32.2 + mlx-lm 0.31.3
(different GatedDeltaNet kernel paths — see Numerics note) the same hardware measures:*

| Prompt | plain tok/s | MTP-v2 tok/s | v2 accept % |
|---|---|---|---|
| France (factual) | 26.2 | 23.3 | 59.5% |
| ML (expository) | 24.2 | 23.2 | 56.7% |
| Genesis (structured) | 26.2 | **34.3** | 85.2% |
| average | 25.6 | 26.9 | — |

*Rule of thumb: MTP-v2 pays off on structured/high-acceptance content (code, lists,
liturgy, boilerplate) and is neutral-to-slightly-negative on diverse prose. Run
`tools/probe_batch_numerics.py <model-dir>` to characterize your own stack.*

MTP acceptance rate: 66–86%, content-dependent (higher on structured / repetitive text).
MTP-v2 emits only verified-argmax tokens — every token accepted is `argmax` of a full
model forward, conditioned on a valid prior context. Output quality is equivalent to
plain greedy. Due to a GatedDeltaNet recurrent-state accumulation difference between
T=1 and T>1 kernel paths (see Numerics note below), the token sequence may diverge from
plain greedy at positions where accumulated state drift tips an argmax near-tie.

MTP-v1 (sequential mode, `--mtp`) exists but is slower than plain greedy. Use `--mtp-v2`.

---

## Requirements

```
mlx >= 0.26.0
mlx-lm >= 0.21.0
numpy >= 1.24
transformers >= 4.45
```

Install:

```bash
pip install mlx mlx-lm numpy transformers
```

---

## Quickstart

### 1. Download the 4-bit weights

```bash
huggingface-cli download RockTalk/Qwen3.8-Flash-Next-MLX-4bit \
    --local-dir /path/to/model-dir
```

The download is ~71 GB (49 safetensors shards + mtp-weights.safetensors + tokenizer files).
The n-gram table (`ngram_table.bin`, ~95 GB) is a separate optional download; without it
the model runs in degraded mode (PLE skipped, one warning printed, output still coherent).

### 2. Generate

```bash
python3 tools/run_mlx.py --model-dir /path/to/model-dir \
    --prompt "The capital of France is" --max-tokens 64
```

With MTP-v2 speculative decoding (recommended):

```bash
python3 tools/run_mlx.py --model-dir /path/to/model-dir \
    --prompt "The capital of France is" --max-tokens 64 --mtp-v2
```

Chat mode:

```bash
python3 tools/run_mlx.py --model-dir /path/to/model-dir \
    --prompt "Explain Rayleigh scattering in one paragraph." --chat --max-tokens 200
```

Alternatively set `QWEN_MODEL_DIR` and omit `--model-dir`:

```bash
export QWEN_MODEL_DIR=/path/to/model-dir
python3 tools/run_mlx.py --prompt "The capital of France is"
```

### 3. Run the equality / benchmark test

```bash
# Equality test (checks MTP-v2 vs plain greedy token-for-token)
python3 tools/run_mlx.py --model-dir /path/to/model-dir --equality-test

# Benchmark (plain vs MTP-v2 speed comparison)
python3 tools/run_mlx.py --model-dir /path/to/model-dir --benchmark
```

> **Numerics note:** mlx-lm's GatedDeltaNet implementation uses different kernel paths
> for T=1 (single-step) and T>1 (batched) forwards. The T=2 batch-verify step produces
> logits that argmax-match the equivalent sequential T=1 step, but the internal
> recurrent state written to `ArraysCache` after a T=2 forward differs from the state
> produced by two successive T=1 forwards. This accumulated state divergence causes
> the MTP-v2 token sequence to diverge from plain greedy after several accept/reject
> cycles. The divergence is not a decode error — every accepted token is the correct
> argmax of a full forward — but the subsequent context state differs, so completions
> may differ.
>
> `--equality-test` will report FAIL on mlx 0.32.x. On mlx ≤ 0.31.x the T=2 path
> reduces to sequential single steps and the test passes. This is a known limitation;
> a fix (regenerating the T=1 cache state after each accept) would eliminate the speedup.
> Use `--benchmark` to verify speed on your stack.

---

## Building weights from the original Qwen checkpoint

If you want to build the MLX weights yourself from Qwen's BF16 checkpoint:

### Step 1 — Convert BF16 → MLX bf16

```bash
python3 -m mlx_qwen4exp.convert \
    --hf-dir /path/to/Qwen3.8-Flash-Next-BF16 \
    --out-dir /path/to/MLX-bf16 \
    --skip-table   # omit to also build the 95 GB n-gram table
```

The converter is restartable: each shard is written atomically and skipped on re-run.
Use `--limit-shards N` to smoke-test the pipeline on the first N shards.

### Step 2 — Quantize to 4-bit

```bash
python3 tools/quantize_stream.py \
    --src-dir /path/to/MLX-bf16 \
    --dst-dir /path/to/MLX-4bit
```

Peak RSS during quantization: ~11 GB. Wall time: ~55 s on M3 Ultra. Output: ~71 GB.

### Step 3 — Add MTP weights (optional)

```bash
python3 tools/quantize_mtp.py \
    --bf16-dir /path/to/Qwen3.8-Flash-Next-BF16 \
    --dst-dir  /path/to/MLX-4bit
```

This reads mtp.* tensors from the BF16 checkpoint, sanitizes, quantizes, and injects
them as `mtp-weights.safetensors` into the 4-bit directory.

---

## Running tests

```bash
pip install pytest
python3 -m pytest tests/ -v
```

110 tests, all passing. Coverage includes:

| Module | Tests | What is verified |
|--------|-------|-----------------|
| hyper.py | 31 | 4-stream gated residual, grouped-norm vs reference loop, mutation tests |
| ple.py | 14 | n-gram hash bit-exact vs compiled C, chunked==single-shot, conv dilations |
| deltanet.py | 7 | `1/sqrt(head_v_dim)` scale, sigmoid gate, chunked==single-shot |
| moe.py | 8 | routing matches dense reference, top-k renorm, shared expert gate |
| attention.py | 10 | dense==sparse equivalence ≤2051 tokens, incremental==prefill |
| model.py | 10 | strict-load, +1 gamma folding, sanitize table, 48-layer toy forward |
| mtp.py | 27 | forward shape, cache wiring, degenerate accept-all==greedy, snapshot/restore |

The degenerate equivalence test (`TestDegenerateEquivalence`) is the constructed-equivalence
leg: when the draft is always accepted, speculative decode must produce the same tokens as
plain greedy. This is the correctness proof for the speculative loop itself.

---

## Limitations

- **Text only**: the model ships a ViT for multimodal input; this port does not implement
  the vision path. Text-only inference is complete.
- **Sparse mask O(n²) beyond 2051 tokens**: the QSA sparse attention mask is computed as
  a dense `[n_kv, n_kv]` matrix. Above 2051 tokens the mask correctly selects a budget
  subset but still materializes the full matrix. For 262K-token context, this would need
  an argpartition-based sparse gather path.
- **MTP-v1 is slower than plain greedy**: the sequential draft path (`--mtp`) has higher
  overhead than the batch-verify path (`--mtp-v2`). Use `--mtp-v2`.
- **4-bit quantization sensitivity**: the 512 SwitchLinear experts are the quality
  bottleneck. Higher bit-widths (6-bit, 8-bit) improve quality but have no fast MLX
  gather-matmul kernel, resulting in 0.3–0.4 tok/s. 4-bit g64 is the only serving-viable
  MLX quantization today.

---

## License

Code: MIT (see LICENSE).

Dependencies: mlx and mlx-lm are Apache 2.0
([ml-explore/mlx](https://github.com/ml-explore/mlx),
[ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)).

Model weights: Qwen Community License 1.0
([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
