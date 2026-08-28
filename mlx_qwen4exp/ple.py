"""PLE n-gram hash embedding block for qwen4_exp (Qwen3.8-Flash-Next).

Two independent pieces live here:

1. `ngram_rows` — the host-side integer hash. It turns a window of token ids into
   `ple_n_heads` row indices into the (320M-row) shared embedding table. This is pure
   NumPy on purpose: the arithmetic is uint64 with wrapping overflow and an xor, neither
   of which MLX gives us cleanly, and it is cheap (16 ints per token).

2. `PLEBlock` — the tensor half: gather, two projections, a per-hc-stream gated dot
   product and a dilated depthwise causal convolution over time.

Ground truth: `llama.cpp/src/models/qwen4exp.cpp`
  - hash  : `llm_graph_input_ple::set_input`      (lines 895-964)
  - block : `llama_model_qwen4exp::graph::build_ple` (lines 1014-1122)
and SPEC.md sections 4.1 (grouped_norm) / 4.2.

The 320M x 160 table is NOT owned by this module. It is passed into `__call__` so the
caller can keep it memory-mapped / sharded / quantised however it likes.
"""

from typing import Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelArgs

__all__ = ["ngram_rows", "grouped_norm", "PLEBlock"]


# ---------------------------------------------------------------------------
# 1. the hash
# ---------------------------------------------------------------------------


def _as_i64(name: str, values, expected: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if expected is not None and arr.shape[0] != expected:
        raise ValueError(f"{name} must have {expected} entries, got {arr.shape[0]}")
    return arr.astype(np.int64, copy=False)


def ngram_rows(
    tokens,
    prev_tokens,
    multipliers,
    offsets,
    vocab_sizes,
    ngram_size: int,
    heads_per_ngram: int,
    eos_id: int,
) -> np.ndarray:
    """Hash each token's n-gram context into `(ngram_size-1) * heads_per_ngram` table rows.

    Args:
        tokens:      int array [B, T] — the current window's token ids.
        prev_tokens: int array [B, ngram_size-1] — the tokens immediately BEFORE this
                     window, oldest-first, with -1 meaning "does not exist". Threading
                     this through is what makes a chunked prefill match a single-shot one.
        multipliers: int64 [>= ngram_size] — `ple_embedding.layer_multipliers`.
        offsets:     int64 [n_heads]       — `ple_embedding.ngram_heads_offsets`.
        vocab_sizes: int64 [n_heads]       — `ple_embedding.ngram_heads_vocab_sizes`.
        ngram_size:  3 for the real model.
        heads_per_ngram: 8 for the real model.
        eos_id:      the token id that cuts a context window.

    Returns:
        int32 array [B, T, n_heads] of absolute row indices into the shared table.

    All hashing arithmetic is numpy uint64 and therefore WRAPS on overflow, exactly as
    the C++ reference does. Python ints are arbitrary precision and would silently
    produce different rows, so nothing here is ever allowed to become a Python int.
    """
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be >= 2, got {ngram_size}")
    if heads_per_ngram < 1:
        raise ValueError(f"heads_per_ngram must be >= 1, got {heads_per_ngram}")

    tok = np.asarray(tokens)
    if tok.ndim != 2:
        raise ValueError(f"tokens must be 2-D [B, T], got shape {tok.shape}")
    tok = tok.astype(np.int64, copy=False)
    B, T = tok.shape

    n_prev = ngram_size - 1
    prev = np.asarray(prev_tokens)
    if prev.ndim != 2 or prev.shape != (B, n_prev):
        raise ValueError(
            f"prev_tokens must have shape ({B}, {n_prev}), got {tuple(prev.shape)}"
        )
    prev = prev.astype(np.int64, copy=False)

    n_heads = n_prev * heads_per_ngram
    mult = _as_i64("multipliers", multipliers)
    if mult.shape[0] < ngram_size:
        raise ValueError(
            f"multipliers must have at least {ngram_size} entries, got {mult.shape[0]}"
        )
    offs = _as_i64("offsets", offsets, n_heads)
    vocab = _as_i64("vocab_sizes", vocab_sizes, n_heads)
    if np.any(vocab <= 0):
        raise ValueError("vocab_sizes must all be > 0")

    rows_u = np.empty((B, T, n_heads), dtype=np.uint64)
    if T == 0:
        return rows_u.astype(np.uint32).view(np.int32)

    eos = np.int64(eos_id)

    # --- build ctx[0..ngram_size-1] -----------------------------------------
    # ctx[0] is the token itself; ctx[s] is the token s positions back.
    # Walking s = 1..n-1: `cut` latches as soon as a slot is missing (-1) or is EOS;
    # once cut, that slot and every older slot read as EOS. ctx[0] being EOS does NOT
    # cut its own context (C++ lines 934-942).
    ctx = np.empty((ngram_size, B, T), dtype=np.int64)
    ctx[0] = tok

    # [B, n_prev + T]: predecessors then the window, so slot s is one contiguous slice.
    stream = np.concatenate([prev, tok], axis=1)
    cut = np.zeros((B, T), dtype=bool)
    for s in range(1, ngram_size):
        start = n_prev - s
        raw = stream[:, start : start + T]
        t = np.where(cut, np.int64(-1), raw)          # `cut ? LLAMA_TOKEN_NULL : prev[...]`
        cut = cut | (t < 0) | (t == eos)
        ctx[s] = np.where(cut, eos, t)

    # --- mix ----------------------------------------------------------------
    # Wrapping uint64 multiply/xor. numpy warns on the intended overflow; silence it.
    with np.errstate(over="ignore", invalid="ignore"):
        mult_u = mult[:ngram_size].astype(np.uint64)
        terms = ctx.astype(np.uint64) * mult_u[:, None, None]   # [n, B, T]

        for n in range(2, ngram_size + 1):
            mixed = terms[0].copy()
            for j in range(1, n):
                mixed = mixed ^ terms[j]
            base = (n - 2) * heads_per_ngram
            v = vocab[base : base + heads_per_ngram].astype(np.uint64)
            o = offs[base : base + heads_per_ngram].astype(np.uint64)
            rows_u[:, :, base : base + heads_per_ngram] = (mixed[:, :, None] % v) + o

    # `(int32_t) value` in C++ == truncate to 32 bits, reinterpret as signed.
    return rows_u.astype(np.uint32).view(np.int32)


# ---------------------------------------------------------------------------
# 2. grouped RMS norm (SPEC 4.1 / qwen4exp.cpp:1043-1049)
# ---------------------------------------------------------------------------


def grouped_norm(x_flat: mx.array, weight: mx.array, hc: int, eps: float) -> mx.array:
    """RMS-normalise per hc stream, then scale the flattened layout by `weight`.

    `x_flat` is [B, T, hc*D]. The normalisation reduces over D on the [B, T, hc, D]
    view (so each of the hc streams is normalised on its own), and only afterwards does
    the [hc*D] gamma multiply the flattened tensor. A plain RMSNorm over hc*D is NOT
    the same thing.
    """
    if x_flat.ndim != 3:
        raise ValueError(f"grouped_norm expects [B, T, hc_dim], got {x_flat.shape}")
    B, T, hc_dim = x_flat.shape
    if hc_dim % hc != 0:
        raise ValueError(f"hc_dim {hc_dim} is not divisible by hc {hc}")
    if weight.shape != (hc_dim,):
        raise ValueError(f"grouped_norm weight must be [{hc_dim}], got {weight.shape}")

    x = x_flat.reshape(B, T, hc, hc_dim // hc)
    x = x * mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + eps).astype(
        x.dtype
    )
    return x.reshape(B, T, hc_dim) * weight


# ---------------------------------------------------------------------------
# 3. the block
# ---------------------------------------------------------------------------


class PLEBlock(nn.Module):
    """Per-layer n-gram embedding block (layer 1, 0-based, in the real model).

    Parameter attribute names match the checkpoint after the `{L}.ple.` prefix is
    stripped, with the three `*.weight` norm gammas flattened onto the module:
        key_proj.weight  value_proj.weight  norm_key  norm_query  norm_conv  conv1d
    `conv1d` is the checkpoint's [hc_dim, 1, K] with the middle axis squeezed away.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc = args.hc_count
        self.hidden_size = args.hidden_size
        self.hc_dim = args.hc_dim
        self.eps = args.rms_norm_eps
        self.n_heads = args.ple_n_heads
        self.head_dim = args.ple_head_dim
        self.embed_dim = args.ple_embed_dim
        self.kernel_size = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.history = args.ple_conv_history  # (K - 1) * dilation

        self.key_proj = nn.Linear(args.ple_embed_dim, args.hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)

        self.norm_key = mx.ones((args.hc_dim,))
        self.norm_query = mx.ones((args.hc_dim,))
        self.norm_conv = mx.ones((args.hc_dim,))

        # depthwise: one weight per (channel, tap). Same fan-in convention as nn.Linear.
        scale = 1.0 / (args.ple_conv_kernel_size ** 0.5)
        self.conv1d = mx.random.uniform(
            low=-scale, high=scale, shape=(args.hc_dim, args.ple_conv_kernel_size)
        )

    def initial_state(self, batch_size: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        """A zeroed conv history, [B, ple_conv_history, hc_dim]."""
        return mx.zeros((batch_size, self.history, self.hc_dim), dtype=dtype)

    def __call__(
        self,
        hidden_wide: mx.array,
        rows: Union[mx.array, np.ndarray],
        table: mx.array,
        conv_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            hidden_wide: [B, T, hc, D]  the wide residual stream.
            rows:        int32 [B, T, ple_n_heads] from `ngram_rows`.
            table:       [n_rows, ple_head_dim]  the shared n-gram embedding table.
            conv_state:  [B, ple_conv_history, hc_dim] or None (zeros).

        Returns:
            (out_wide [B, T, hc, D], new_conv_state [B, ple_conv_history, hc_dim])
        """
        if hidden_wide.ndim != 4:
            raise ValueError(
                f"hidden_wide must be [B, T, hc, D], got {hidden_wide.shape}"
            )
        B, T, hc, D = hidden_wide.shape
        if hc != self.hc or D != self.hidden_size:
            raise ValueError(
                f"hidden_wide must be [B, T, {self.hc}, {self.hidden_size}], "
                f"got {hidden_wide.shape}"
            )

        if isinstance(rows, np.ndarray):
            rows = mx.array(rows.astype(np.int32, copy=False))
        if rows.shape != (B, T, self.n_heads):
            raise ValueError(
                f"rows must be [{B}, {T}, {self.n_heads}], got {rows.shape}"
            )
        if table.ndim != 2 or table.shape[1] != self.head_dim:
            raise ValueError(
                f"table must be [n_rows, {self.head_dim}], got {table.shape}"
            )

        # gather: the head axis is the slow one, so [B,T,H,head_dim] flattens straight
        # into ple_embed_dim (qwen4exp.cpp:1034-1036).
        emb = table[rows].reshape(B, T, self.embed_dim)

        key = self.key_proj(emb)                                   # [B,T,hc_dim]
        value = self.value_proj(emb)                               # [B,T,D]

        key_w = grouped_norm(key, self.norm_key, hc, self.eps).reshape(B, T, hc, D)
        query = grouped_norm(
            hidden_wide.reshape(B, T, self.hc_dim), self.norm_query, hc, self.eps
        ).reshape(B, T, hc, D)

        # per-stream dot product, then a SIGNED square root before the sigmoid
        s = (key_w * query).sum(axis=-1) / (D ** 0.5)              # [B,T,hc]
        gate = mx.sigmoid(mx.sign(s) * mx.sqrt(mx.maximum(mx.abs(s), 1e-6)))
        gated = value[:, :, None, :] * gate[:, :, :, None]         # [B,T,hc,D]

        normalized = grouped_norm(
            gated.reshape(B, T, self.hc_dim), self.norm_conv, hc, self.eps
        )                                                          # [B,T,hc_dim]

        # depthwise causal conv over TIME, kernel K, dilated by ngram_size:
        #   out[b,t,c] = sum_k w[c,k] * xpad[b, t + k*dilation, c]
        # tap k therefore reads (K-1-k)*dilation positions back from t, matching
        # qwen4exp.cpp:1092-1114 (start = hist - (K-1-k)*dil == k*dil).
        if conv_state is None:
            conv_state = self.initial_state(B, normalized.dtype)
        elif conv_state.shape != (B, self.history, self.hc_dim):
            raise ValueError(
                f"conv_state must be [{B}, {self.history}, {self.hc_dim}], "
                f"got {conv_state.shape}"
            )
        xpad = mx.concatenate([conv_state.astype(normalized.dtype), normalized], axis=1)

        conv = None
        for k in range(self.kernel_size):
            start = k * self.dilation
            term = xpad[:, start : start + T, :] * self.conv1d[:, k]
            conv = term if conv is None else conv + term
        conv_out = nn.silu(conv).reshape(B, T, hc, D)

        new_conv_state = xpad[:, xpad.shape[1] - self.history :, :]

        return hidden_wide + gated + conv_out, new_conv_state
