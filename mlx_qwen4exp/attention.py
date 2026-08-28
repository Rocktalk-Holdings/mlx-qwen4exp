"""QSA — Qwen3.8-Flash-Next sparse full-attention layer (the `full_attention` layers).

Ground truth (C++/ggml): ``llama.cpp/src/models/qwen4exp.cpp`` lines 389-691, plus the
sparse-mask host code in ``llama.cpp/src/llama-memory-hybrid-idx.cpp`` lines 352-452.

Two paths through one attention core:

  * DENSE  — plain causal GQA. Exact whenever ``n_kv <= budget + r - 1`` (=2051 at real
             dims), because the top-k of that width selects every cell anyway. Even on the
             dense path the RAW indexer keys are still cached, so a later long-context step
             has the history it needs.
  * SPARSE — indexer scores whole blocks of ``r`` mean-pooled keys, the top ``budget+r-1``
             cells (whole blocks + ragged tail) are kept, and attention runs against a
             causal mask with everything else set to -inf.

Q / GATE interleave (cpp:623-649): ``q_proj`` emits ``[B,T, n_head*2*head_dim]`` laid out
per head as ``[query(head_dim) | gate(head_dim)]``. Viewed as ``[B,T,n_head,2,head_dim]``
the query is index 0 and the gate is index 1 on the size-2 axis. This is the #1
silent-failure risk; ``test_attention.py`` pins the exact stride.

RoPE is PARTIAL: only the first ``n_rot`` (=64 at real dims) of each head's ``head_dim`` is
rotated; the rest passes through. ``mx.fast.rope(x, dims=n_rot, ...)`` does exactly that.
Text-only mrope (all three position ids equal) degenerates to a standard partial RoPE
(SPEC 4.6), so we use a single position stream here and leave true multimodal mrope for when
vision lands.

BLOCK POSITION for the indexer key RoPE: VERIFIED against
``llama-memory-hybrid-idx.cpp:378-386`` (``set_input_qsa``). ``blk_pos`` for block ``b`` is
``b * r`` — the block's FIRST token position — carried identically in all four mrope
sections. Not a guess.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .config import ModelArgs


# --------------------------------------------------------------------------- helpers
def _rms_norm(x: mx.array, gamma: mx.array, eps: float) -> mx.array:
    """RMSNorm over the last axis with an explicit gamma vector."""
    return mx.fast.rms_norm(x, gamma, eps)


# --------------------------------------------------------------------------- indexer cache
class IndexerCache:
    """Holds the RAW indexer keys (pre-norm, pre-rope) for every cached token.

    Mirrors ``mlx_lm.models.cache.KVCache`` (same growth-in-steps discipline) but stores a
    single ``[B, n_kv, idx_dim]`` tensor, since the indexer has one kv head. It must track
    the attention KVCache cell-for-cell: for the same token step, ``update_and_fetch`` is
    called with the same ``T`` and returns the same ``n_kv`` length.
    """

    step = 256

    def __init__(self) -> None:
        self.keys: Optional[mx.array] = None
        self.offset = 0

    def update_and_fetch(self, new: mx.array) -> mx.array:
        # new: [B, T, idx_dim]
        prev = self.offset
        T = new.shape[1]
        if self.keys is None or (prev + T) > self.keys.shape[1]:
            B, _, D = new.shape
            n_steps = (self.step + T - 1) // self.step
            shape = (B, n_steps * self.step, D)
            grow = mx.zeros(shape, new.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[:, :prev, :]
                self.keys = mx.concatenate([self.keys, grow], axis=1)
            else:
                self.keys = grow
        self.offset += T
        self.keys[:, prev : self.offset, :] = new
        return self.keys[:, : self.offset, :]

    @property
    def state(self):
        if self.keys is None:
            return None
        if self.offset == self.keys.shape[1]:
            return self.keys
        return self.keys[:, : self.offset, :]

    @state.setter
    def state(self, v):
        self.keys = v
        self.offset = 0 if v is None else v.shape[1]

    def size(self) -> int:
        return self.offset


# --------------------------------------------------------------------------- indexer
class QSAIndexer(nn.Module):
    """DeepSeek-style lightning indexer that scores whole blocks of pooled keys."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.indexer_n_heads          # 4
        self.head_dim = args.indexer_head_dim         # 128
        self.r = args.indexer_compress_ratio          # 4
        self.eps = args.rms_norm_eps

        # loader splits index_qk_proj [640,2560] -> q rows 0..511, k rows 512..639
        self.index_q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.index_k_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        # zero-centred gammas (loader adds 1.0); [128]
        self.q_layernorm = mx.ones((self.head_dim,))
        self.k_layernorm = mx.ones((self.head_dim,))

        base = args.rope_theta
        self.rope = nn.RoPE(dims=self.head_dim, traditional=False, base=base)

    def raw_keys(self, x: mx.array) -> mx.array:
        """RAW indexer keys for this step: [B, T, idx_dim]. Cached BEFORE norm+rope."""
        return self.index_k_proj(x)

    def block_scores(
        self, x: mx.array, k_raw_all: mx.array, pos_offset: int
    ) -> Tuple[mx.array, int]:
        """Return (token_block_score[B,T,n_blocks_expanded], n_kv) — pre-bias.

        k_raw_all: [B, n_kv, idx_dim] full cache of raw keys.
        The score of block j is broadcast; the caller maps blocks -> token cells.
        """
        B, T, _ = x.shape
        n_kv = k_raw_all.shape[1]
        r = self.r
        n_blocks = (n_kv + r - 1) // r

        # --- pool the raw keys into blocks (ragged tail = mean of its members) ---
        pad = n_blocks * r - n_kv
        if pad:
            # pad with zeros then correct the tail block's divisor
            k_pad = mx.concatenate(
                [k_raw_all, mx.zeros((B, pad, self.head_dim), k_raw_all.dtype)], axis=1
            )
        else:
            k_pad = k_raw_all
        k_blk = k_pad.reshape(B, n_blocks, r, self.head_dim)
        summed = k_blk.sum(axis=2)                                   # [B, n_blocks, D]
        counts = mx.full((n_blocks,), float(r), dtype=summed.dtype)
        if pad:
            counts = counts.at[n_blocks - 1].add(-float(pad))
        pooled = summed / counts[None, :, None]                      # [B, n_blocks, D]

        # --- norm then rope with block position = b*r (VERIFIED cpp set_input_qsa) ---
        pooled = _rms_norm(pooled, self.k_layernorm, self.eps)
        blk_pos = mx.arange(n_blocks) * r                            # [n_blocks]
        pooled = self._rope_at(pooled[:, :, None, :], blk_pos)[:, :, 0, :]  # [B,n_blocks,D]

        # --- query: [B,T,n_heads,D], norm, rope at token positions ---
        q = self.index_q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        q = _rms_norm(q, self.q_layernorm, self.eps)
        q_pos = mx.arange(pos_offset, pos_offset + T)
        q = self._rope_at(q, q_pos)                                  # [B,T,n_heads,D]

        # --- score = relu(q . pooled) then SUM over heads (relu BEFORE sum, cpp:494-501) ---
        # q: [B,T,H,D]  pooled: [B,nb,D]  ->  [B,T,H,nb]
        scores = mx.einsum("bthd,bnd->bthn", q, pooled)
        scores = mx.maximum(scores, 0.0)
        block_score = scores.sum(axis=2)                            # [B, T, n_blocks]
        return block_score, n_kv

    def _rope_at(self, x: mx.array, positions: mx.array) -> mx.array:
        """Apply partial RoPE where each token/block along axis 1 has an explicit position.

        x: [B, L, H, D]. positions: [L]. Because ``mx.fast.rope`` only accepts a scalar
        offset, and our positions are contiguous (arange-based) in every call site, we rope
        with the group's base offset (positions[0]) — contiguity makes that exact.
        """
        B, L, H, D = x.shape
        base = int(positions[0].item()) if L > 0 else 0
        xr = x.transpose(0, 2, 1, 3)          # [B,H,L,D]
        xr = self.rope(xr, offset=base)
        return xr.transpose(0, 2, 1, 3)       # [B,L,H,D]


# --------------------------------------------------------------------------- attention
class QSAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads       # 24
        self.n_kv_heads = args.num_key_value_heads     # 2
        self.head_dim = args.head_dim                  # 256
        self.n_rot = args.n_rot                        # 64
        self.eps = args.rms_norm_eps
        self.scale = 1.0 / math.sqrt(self.head_dim)

        H, KV, HD = self.n_heads, self.n_kv_heads, self.head_dim
        # query||gate interleaved per head => 2*HD per head
        self.q_proj = nn.Linear(args.hidden_size, H * 2 * HD, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, KV * HD, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, KV * HD, bias=False)
        self.o_proj = nn.Linear(H * HD, args.hidden_size, bias=False)

        # RMSNorm gammas over head_dim (standard Qwen rule, no +1)
        self.q_norm = mx.ones((HD,))
        self.k_norm = mx.ones((HD,))

        self.rope = nn.RoPE(dims=self.n_rot, traditional=False, base=args.rope_theta)

        self.indexer = QSAIndexer(args)

        # dense == sparse threshold: below this, top-k selects everything (cpp:514)
        self.dense_threshold = args.indexer_budget + args.indexer_compress_ratio - 1

    # ---------------------------------------------------------------- sparse mask
    def _sparse_mask(
        self,
        block_score: mx.array,   # [B, T, n_blocks]
        n_kv: int,
        pos_offset: int,
    ) -> mx.array:
        """Additive [B, 1, T, n_kv] mask: 0 for kept cells, -inf elsewhere.

        LANDMINE / PERF: the exact-width tie-break builds an [B,T,n_kv,n_kv] outrank tensor.
        That is fine at test/mid dims but is O(n_kv^2) memory and will NOT scale to the full
        262K context. For production long-context, replace this rank matrix with an
        argsort/top-k gather (mx has no ggml_top_k equivalent that returns indices cheaply in
        0.30.6; a chunked argpartition is the intended follow-up). Correctness-first here.


        Reproduces ``set_input_qsa`` bias + the top-k unmask, combined with causality:
          * future cells (pos > q): -inf (causal)
          * tail cells (pos >= tail_start): forced in with bias 1e9  (always survive top-k)
          * complete-block cells: bias 0
          * incomplete non-tail block cells: -inf
        then keep the top ``width = min(n_kv, budget+r-1)`` cells per query row (0), mask
        the rest (-inf).
        """
        B, T, n_blocks = block_score.shape
        r = self.args.indexer_compress_ratio

        # map each cell j -> its block score  ->  token_score[b,t,j]
        cell_block = mx.arange(n_kv) // r                       # [n_kv]
        # gather block scores per cell: [B,T,n_kv]
        tok_score = mx.take(block_score, cell_block, axis=2)    # [B,T,n_kv]

        # positions
        j_pos = mx.arange(n_kv)[None, None, :]                  # [1,1,n_kv]
        q_pos = mx.arange(pos_offset, pos_offset + T)[None, :, None]  # [1,T,1]
        tail_start = ((q_pos + 1) // r) * r                     # [1,T,1]

        causal = j_pos <= q_pos                                 # visible under causality
        is_tail = j_pos >= tail_start                           # ragged tail of q's block

        # completeness of block b: which blocks have all r members present in [0,n_kv).
        # For a contiguous cache every block except possibly the last is complete; the last
        # block is complete iff n_kv % r == 0. Matches the "filled[b] == r" test in C++.
        full_blocks = (n_kv // r) * r
        cell_complete = (mx.arange(n_kv) < full_blocks)[None, None, :]  # [1,1,n_kv]

        NEG = -1e30
        # bias: tail -> big; complete visible -> 0; incomplete non-tail visible -> -inf
        bias = mx.where(
            causal,
            mx.where(is_tail, 1e9, mx.where(cell_complete, 0.0, NEG)),
            NEG,
        )                                                      # [B?,T,n_kv] broadcast
        bias = mx.broadcast_to(bias, (B, T, n_kv))
        combined = tok_score + bias                            # [B,T,n_kv]

        width = min(n_kv, self.dense_threshold)
        if width >= n_kv:
            # everything selected -> pure causal mask
            keep = mx.broadcast_to(causal, (B, T, n_kv))
        else:
            # Match ggml_top_k: keep EXACTLY `width` cells per query row. Ties (two cells
            # with equal combined score) break toward the LOWER cell index, matching the
            # reference's stable index order. We do this magnitude-independently via a rank:
            # cell j is kept iff strictly fewer than `width` OTHER cells outrank it, where the
            # ordering is (score DESC, index ASC).
            valid = combined > NEG / 2                              # [B,T,n_kv] selectable
            c_i = combined[:, :, :, None]                          # [B,T,n_kv,1]
            c_j = combined[:, :, None, :]                          # [B,T,1,n_kv]
            idx = mx.arange(n_kv)
            i_lt_j = (idx[:, None] > idx[None, :])                 # [n_kv,n_kv] j has lower idx
            # j outranks i  <=>  score_j > score_i, or equal score and j is the lower index
            outranks = (c_j > c_i) | ((c_j == c_i) & i_lt_j[None, None, :, :])
            # only valid cells can outrank; and only valid cells can be kept
            outranks = outranks & valid[:, :, None, :]
            rank = outranks.sum(axis=-1)                           # [B,T,n_kv] #cells above i
            keep = (rank < width) & valid

        mask = mx.where(keep, 0.0, NEG).astype(mx.float32)      # [B,T,n_kv]
        return mask[:, None, :, :]                              # [B,1,T,n_kv]

    # ---------------------------------------------------------------- forward
    def __call__(
        self,
        x: mx.array,
        cache=None,
        idx_cache: Optional[IndexerCache] = None,
        pos_offset: int = 0,
    ) -> mx.array:
        B, T, _ = x.shape
        H, KV, HD = self.n_heads, self.n_kv_heads, self.head_dim

        # --- q/gate interleave: [B,T,H,2,HD]; q=index0, gate=index1 (cpp:623-649) ---
        qg = self.q_proj(x).reshape(B, T, H, 2, HD)
        q = qg[:, :, :, 0, :]                                   # [B,T,H,HD]
        gate = qg[:, :, :, 1, :]                                # [B,T,H,HD]

        k = self.k_proj(x).reshape(B, T, KV, HD)
        v = self.v_proj(x).reshape(B, T, KV, HD)

        # --- head-dim RMSNorm on q,k ---
        q = _rms_norm(q, self.q_norm, self.eps)
        k = _rms_norm(k, self.k_norm, self.eps)

        # --- partial RoPE (first n_rot dims), positions offset by pos_offset ---
        q = q.transpose(0, 2, 1, 3)                             # [B,H,T,HD]
        k = k.transpose(0, 2, 1, 3)                             # [B,KV,T,HD]
        v = v.transpose(0, 2, 1, 3)                             # [B,KV,T,HD]
        q = self.rope(q, offset=pos_offset)
        k = self.rope(k, offset=pos_offset)

        # --- cache the RAW indexer keys ALWAYS (even on dense path) ---
        raw = self.indexer.raw_keys(x)                          # [B,T,idx_dim]
        if idx_cache is not None:
            k_raw_all = idx_cache.update_and_fetch(raw)         # [B,n_kv,idx_dim]
        else:
            k_raw_all = raw

        # --- k/v cache ---
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)                # [B,KV,n_kv,HD]

        n_kv = k.shape[2]

        # --- choose mask: dense causal, or sparse restricted ---
        if n_kv <= self.dense_threshold:
            # dense path: plain causal mask. (Indexer keys already cached above.)
            mask = self._causal_mask(T, n_kv, pos_offset)
        else:
            block_score, nkv2 = self.indexer.block_scores(x, k_raw_all, pos_offset)
            mask = self._sparse_mask(block_score, n_kv, pos_offset)

        # SDPA requires the additive mask to promote to the query/value dtype. The mask
        # builders emit float32 (for headroom on the -1e30 fill); with a bf16 residual
        # stream (e.g. quantized weights) float32 does NOT promote down to bf16, so cast
        # the mask to q.dtype here. bf16 min is ~-3.4e38, so -1e30 stays a hard mask.
        if mask.dtype != q.dtype:
            mask = mask.astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )                                                      # [B,H,T,HD]

        # --- per-head sigmoid gate BEFORE o_proj (cpp:681-687) ---
        out = out.transpose(0, 2, 1, 3)                        # [B,T,H,HD]
        out = out * mx.sigmoid(gate)
        out = out.reshape(B, T, H * HD)
        return self.o_proj(out)

    def _causal_mask(self, T: int, n_kv: int, pos_offset: int) -> mx.array:
        q_pos = mx.arange(pos_offset, pos_offset + T)[:, None]   # [T,1]
        j_pos = mx.arange(n_kv)[None, :]                          # [1,n_kv]
        keep = j_pos <= q_pos
        return mx.where(keep, 0.0, -1e30).astype(mx.float32)[None, None, :, :]


# --------------------------------------------------------------------------- QSA cache
class QSACache:
    """A small pair-holder so mlx_lm.generate (which passes cache[i] to layer i)
    hands a QSA layer both its KVCache and its IndexerCache in one object.

    The model unpacks .kv / .idx itself; mlx_lm only ever calls make_cache on
    the model and then indexes the returned list, so a plain holder is enough. .offset
    proxies the KVCache offset, which is what mlx_lm reads to size masks / positions.

    Placed in attention.py (not model.py) to avoid circular imports with mtp.py.
    """

    def __init__(self) -> None:
        self.kv = KVCache()
        self.idx = IndexerCache()

    @property
    def offset(self) -> int:
        return self.kv.offset

    @property
    def state(self):
        # mlx_lm KVCache.state crashes if keys is None (mlx_lm bug in some versions).
        # Guard both caches: return None for uninitialized caches.
        try:
            kv_state = self.kv.state
        except (AttributeError, TypeError):
            kv_state = None
        try:
            idx_state = self.idx.state
        except (AttributeError, TypeError):
            idx_state = None
        return (kv_state, idx_state)

    @state.setter
    def state(self, v):
        self.kv.state, self.idx.state = v
