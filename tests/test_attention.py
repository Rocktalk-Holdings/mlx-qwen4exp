"""Tests for mlx_qwen4exp.attention — the QSA full-attention layer.

Toy dimensions (real-ish, tiny so the sparse path triggers at small n_kv):
    hidden 64, heads 4, kv_heads 2, head_dim 16, n_rot 4,
    indexer 2 heads x 8, budget 8, ratio 2  =>  dense/sparse threshold = 9.

No checkpoint required; all weights are random. Every value assertion forces computation
via np.array(...) / float(...); a NaN produced anywhere surfaces there.

Run:
    /opt/homebrew/bin/python3.11 -m pytest tests/test_attention.py -q
"""

import sys
from pathlib import Path

import numpy as np

# repo root on sys.path so `mlx_qwen4exp` (implicit namespace package) imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mlx_qwen4exp.config import ModelArgs  # noqa: E402
from mlx_qwen4exp.attention import QSAAttention, IndexerCache  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402


def toy_args(**over):
    base = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.25,  # n_rot = 16*0.25 = 4
        rope_theta=1e7,
        rms_norm_eps=1e-6,
        indexer_n_heads=2,
        indexer_head_dim=8,
        indexer_kv_heads=1,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )
    base.update(over)
    return ModelArgs(**base)


def build(args=None, seed=0):
    if args is None:
        args = toy_args()
    mx.random.seed(seed)
    attn = QSAAttention(args)

    def randomize(p):
        return mx.random.normal(p.shape) * 0.1

    attn.update(tree_map(randomize, attn.parameters()))
    return attn, args


# --------------------------------------------------------------------------- interleave
def test_q_gate_interleave_stride():
    """q lands at head offset [h*2*hd .. +hd), gate at [h*2*hd+hd .. (h+1)*2*hd)."""
    args = toy_args()
    attn = QSAAttention(args)
    H, HD = args.num_attention_heads, args.head_dim
    out_dim = H * 2 * HD  # 4*2*16 = 128

    # weight so that q_proj(x) == arange(out_dim) for x = e_0 (first hidden unit = 1)
    # nn.Linear computes x @ W.T ; set W[:,0] = arange, rest 0.
    W = np.zeros((out_dim, args.hidden_size), dtype=np.float32)
    W[:, 0] = np.arange(out_dim, dtype=np.float32)
    attn.q_proj.weight = mx.array(W)

    x = np.zeros((1, 1, args.hidden_size), dtype=np.float32)
    x[0, 0, 0] = 1.0
    qg = attn.q_proj(mx.array(x)).reshape(1, 1, H, 2, HD)
    qg = np.array(qg)[0, 0]  # [H,2,HD]

    for h in range(H):
        q_expect = np.arange(h * 2 * HD, h * 2 * HD + HD, dtype=np.float32)
        g_expect = np.arange(h * 2 * HD + HD, (h + 1) * 2 * HD, dtype=np.float32)
        assert np.array_equal(qg[h, 0], q_expect), (h, qg[h, 0], q_expect)
        assert np.array_equal(qg[h, 1], g_expect), (h, qg[h, 1], g_expect)


# --------------------------------------------------------------------------- shapes / GQA
def test_gqa_shapes_no_cache():
    attn, args = build()
    x = mx.random.normal((2, 5, args.hidden_size))
    out = attn(x, cache=None, idx_cache=None, pos_offset=0)
    assert out.shape == (2, 5, args.hidden_size)
    assert not np.isnan(np.array(out)).any()


def test_gqa_shapes_with_cache():
    attn, args = build()
    x = mx.random.normal((1, 6, args.hidden_size))
    out = attn(x, cache=KVCache(), idx_cache=IndexerCache(), pos_offset=0)
    assert out.shape == (1, 6, args.hidden_size)
    assert not np.isnan(np.array(out)).any()


# --------------------------------------------------------------------------- partial rope
def test_partial_rope_passes_tail_through():
    """RoPE (dims=n_rot) leaves dims >= n_rot unchanged; dims < n_rot vary with position."""
    args = toy_args()
    attn = QSAAttention(args)
    HD, n_rot = args.head_dim, args.n_rot
    # a single head, two positions, distinct per-dim signal
    x = mx.broadcast_to(
        mx.arange(HD).astype(mx.float32)[None, None, None, :], (1, 1, 2, HD)
    )
    r0 = np.array(attn.rope(x, offset=0))[0, 0]  # [2,HD]  pos 0
    r5 = np.array(attn.rope(x, offset=5))[0, 0]  # [2,HD]  pos 5 for first row
    inp = np.array(x)[0, 0, 0]
    # tail dims (>= n_rot) identical to input and across offsets
    assert np.allclose(r0[0, n_rot:], inp[n_rot:], atol=1e-6)
    assert np.allclose(r5[0, n_rot:], inp[n_rot:], atol=1e-6)
    # rotated dims vary with position
    assert not np.allclose(r0[0, :n_rot], r5[0, :n_rot], atol=1e-4)


# --------------------------------------------------------------------------- dense==sparse
def _force_sparse_mask(attn, x, n_kv, pos_offset, k_raw_all):
    """Directly build the sparse mask (bypassing the n_kv gate) for equivalence checks."""
    block_score, _ = attn.indexer.block_scores(x, k_raw_all, pos_offset)
    return attn._sparse_mask(block_score, n_kv, pos_offset)


def test_dense_equals_sparse_mask_when_all_selected():
    """budget+r-1 >= n_kv => sparse mask == plain causal mask."""
    args = toy_args(indexer_budget=8, indexer_compress_ratio=2)  # threshold 9
    attn, _ = build(args)
    T = 7  # n_kv = 7 <= 9
    x = mx.random.normal((1, T, args.hidden_size))
    raw = attn.indexer.raw_keys(x)
    mask_sparse = _force_sparse_mask(attn, x, T, 0, raw)
    mask_dense = attn._causal_mask(T, T, 0)
    assert np.allclose(np.array(mask_sparse), np.array(mask_dense), atol=0)


def test_dense_equals_sparse_full_forward():
    """Full forward: with threshold >= n_kv, forcing the sparse mask == dense path."""
    args = toy_args(indexer_budget=8, indexer_compress_ratio=2)
    attn, _ = build(args, seed=3)
    T = 7
    x = mx.random.normal((1, T, args.hidden_size))
    dense = attn(x, cache=None, idx_cache=None, pos_offset=0)

    B = 1
    H, KV, HD = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    qg = attn.q_proj(x).reshape(B, T, H, 2, HD)
    q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :]
    k = attn.k_proj(x).reshape(B, T, KV, HD)
    v = attn.v_proj(x).reshape(B, T, KV, HD)
    q = mx.fast.rms_norm(q, attn.q_norm, args.rms_norm_eps)
    k = mx.fast.rms_norm(k, attn.k_norm, args.rms_norm_eps)
    q = attn.rope(q.transpose(0, 2, 1, 3), offset=0)
    k = attn.rope(k.transpose(0, 2, 1, 3), offset=0)
    v = v.transpose(0, 2, 1, 3)
    raw = attn.indexer.raw_keys(x)
    mask = _force_sparse_mask(attn, x, T, 0, raw)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3) * mx.sigmoid(gate)
    sparse = attn.o_proj(out.reshape(B, T, H * HD))

    assert np.allclose(np.array(dense), np.array(sparse), atol=1e-5), (
        np.abs(np.array(dense) - np.array(sparse)).max()
    )


# --------------------------------------------------------------------------- sparse restricts
def test_sparse_restricts_and_differs():
    """threshold < n_kv: each query row keeps exactly `width` non-inf cells (bounded by
    causality), and the sparse output differs from a dense (full-causal) output."""
    # threshold = budget + r - 1 = 2 + 2 - 1 = 3
    args = toy_args(indexer_budget=2, indexer_compress_ratio=2)
    attn, _ = build(args, seed=5)
    T = 10  # n_kv = 10 > 3
    x = mx.random.normal((1, T, args.hidden_size))

    raw = attn.indexer.raw_keys(x)
    mask = np.array(_force_sparse_mask(attn, x, T, 0, raw))[0, 0]  # [T, n_kv]

    width = min(T, args.indexer_budget + args.indexer_compress_ratio - 1)  # 3
    for t in range(T):
        row = mask[t]
        n_visible = t + 1
        kept = int((row > -1e29).sum())
        expected = min(width, n_visible)
        assert kept == expected, (t, kept, expected, row)

    sparse_out = np.array(attn(x, cache=None, idx_cache=None, pos_offset=0))

    B = 1
    H, KV, HD = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    qg = attn.q_proj(x).reshape(B, T, H, 2, HD)
    q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :]
    k = attn.k_proj(x).reshape(B, T, KV, HD)
    v = attn.v_proj(x).reshape(B, T, KV, HD)
    q = mx.fast.rms_norm(q, attn.q_norm, args.rms_norm_eps)
    k = mx.fast.rms_norm(k, attn.k_norm, args.rms_norm_eps)
    q = attn.rope(q.transpose(0, 2, 1, 3), offset=0)
    k = attn.rope(k.transpose(0, 2, 1, 3), offset=0)
    v = v.transpose(0, 2, 1, 3)
    dmask = attn._causal_mask(T, T, 0)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=dmask)
    o = o.transpose(0, 2, 1, 3) * mx.sigmoid(gate)
    dense_out = np.array(attn.o_proj(o.reshape(B, T, H * HD)))

    assert not np.allclose(sparse_out, dense_out, atol=1e-4)
    assert not np.isnan(sparse_out).any()


# --------------------------------------------------------------------------- incremental decode
def test_incremental_decode_matches_prefill():
    """Prefill 20, decode 5 one-at-a-time == single 25-token prefill at those positions."""
    args = toy_args()  # threshold 9; 25 tokens exercises the sparse path
    attn, _ = build(args, seed=7)
    T = 25
    x = mx.random.normal((1, T, args.hidden_size))

    full = np.array(attn(x, cache=KVCache(), idx_cache=IndexerCache(), pos_offset=0))

    kv, ic = KVCache(), IndexerCache()
    outs = [np.array(attn(x[:, :20, :], cache=kv, idx_cache=ic, pos_offset=0))]
    for t in range(20, 25):
        outs.append(
            np.array(attn(x[:, t : t + 1, :], cache=kv, idx_cache=ic, pos_offset=t))
        )
    chunked = np.concatenate(outs, axis=1)

    assert chunked.shape == full.shape
    assert np.allclose(full, chunked, atol=1e-4), np.abs(full - chunked).max()
    assert not np.isnan(chunked).any()


# --------------------------------------------------------------------------- no NaN sweep
def test_no_nan_various_lengths():
    attn, args = build(seed=11)
    for T in (1, 3, 9, 10, 25, 40):
        x = mx.random.normal((1, T, args.hidden_size))
        out = np.array(attn(x, cache=None, idx_cache=None, pos_offset=0))
        assert not np.isnan(out).any(), T
        assert not np.isinf(out).any(), T


# --------------------------------------------------------------------------- indexer cache tracks kv
def test_indexer_cache_tracks_kv():
    """IndexerCache offset must equal KVCache offset step-for-step."""
    attn, args = build()
    kv, ic = KVCache(), IndexerCache()
    x = mx.random.normal((1, 20, args.hidden_size))
    attn(x, cache=kv, idx_cache=ic, pos_offset=0)
    assert ic.size() == kv.offset == 20
    attn(mx.random.normal((1, 1, args.hidden_size)), cache=kv, idx_cache=ic, pos_offset=20)
    assert ic.size() == kv.offset == 21
