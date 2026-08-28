"""Tests for the PLE n-gram hash embedding block.

Toy dimensions throughout — nothing here needs the 355 GB checkpoint.
    hidden_size 8, hc_count 4        => hc_dim 32
    ngram_size 3, heads_per_ngram 2  => ple_n_heads 4
    ple_embed_dim 8                  => ple_head_dim 2
    ple_conv_kernel_size 4           => ple_conv_history (4-1)*3 = 9

The test that matters most is `test_ngram_rows_matches_naive_reference`: the hash is
reimplemented below with explicit Python loops over np.uint64 scalars, deliberately
written in a different style from the vectorised implementation, and the two must agree
bit for bit.

Note: nothing calls mx.eval() — every assertion goes through np.array(...)/.shape, both
of which force the MLX graph anyway.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from mlx_qwen4exp.config import ModelArgs  # noqa: E402
from mlx_qwen4exp.ple import PLEBlock, grouped_norm, ngram_rows  # noqa: E402


# ---------------------------------------------------------------------------
# toy fixtures
# ---------------------------------------------------------------------------

EOS = 7
VOCAB = 40
NGRAM = 3
HEADS_PER_NGRAM = 2
N_HEADS = (NGRAM - 1) * HEADS_PER_NGRAM  # 4

# 4 heads, 120 rows each, contiguous => 480 rows total (~500 as specified)
HEAD_VOCAB = np.array([120, 120, 120, 120], dtype=np.int64)
HEAD_OFFSETS = np.array([0, 120, 240, 360], dtype=np.int64)
N_ROWS = int((HEAD_OFFSETS + HEAD_VOCAB).max())  # 480

# deliberately huge, so the uint64 multiply overflows and must wrap
MULTIPLIERS = np.array(
    [6364136223846793005, -4232172114205192657, 1181783497276652981],
    dtype=np.int64,
)


def toy_args(**overrides) -> ModelArgs:
    kwargs = dict(
        hidden_size=8,
        hc_count=4,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        ngram_size=NGRAM,
        heads_per_ngram=HEADS_PER_NGRAM,
        ple_embed_dim=8,
        ple_conv_kernel_size=4,
        eos_token_id=EOS,
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


def call_rows(tokens, prev_tokens):
    return ngram_rows(
        tokens,
        prev_tokens,
        MULTIPLIERS,
        HEAD_OFFSETS,
        HEAD_VOCAB,
        NGRAM,
        HEADS_PER_NGRAM,
        EOS,
    )


# ---------------------------------------------------------------------------
# a separate, deliberately naive reference implementation
# ---------------------------------------------------------------------------


def naive_ngram_rows(tokens, prev_tokens, multipliers, offsets, vocab_sizes,
                     ngram_size, heads_per_ngram, eos_id):
    """Straight transcription of qwen4exp.cpp:930-956. Scalar np.uint64 only."""
    tokens = np.asarray(tokens, dtype=np.int64)
    prev_tokens = np.asarray(prev_tokens, dtype=np.int64)
    B, T = tokens.shape
    n_prev = ngram_size - 1
    n_heads = n_prev * heads_per_ngram
    out = np.zeros((B, T, n_heads), dtype=np.int32)

    def u64(v):
        return np.int64(v).astype(np.uint64)

    for b in range(B):
        for i in range(T):
            # ctx[0] is the token itself; ctx[s] is the token s positions back.
            ctx = [0] * ngram_size
            ctx[0] = int(tokens[b, i])
            cut = False
            for s in range(1, ngram_size):
                if cut:
                    t = -1
                elif i - s >= 0:
                    t = int(tokens[b, i - s])
                else:
                    # prev_tokens is oldest-first: (n_prev - s) + i counts back from
                    # the window start, negative means "before the cache", i.e. missing
                    j = n_prev - s + i
                    t = int(prev_tokens[b, j]) if 0 <= j < n_prev else -1
                cut = cut or t < 0 or t == eos_id
                ctx[s] = eos_id if cut else t

            with np.errstate(over="ignore"):
                for n in range(2, ngram_size + 1):
                    mixed = u64(ctx[0]) * u64(multipliers[0])
                    for j in range(1, n):
                        mixed = mixed ^ (u64(ctx[j]) * u64(multipliers[j]))
                    base = (n - 2) * heads_per_ngram
                    for g in range(heads_per_ngram):
                        h = base + g
                        val = (mixed % u64(vocab_sizes[h])) + u64(offsets[h])
                        out[b, i, h] = np.int32(val.astype(np.uint32).astype(np.int64))
    return out


def naive_call(tokens, prev_tokens):
    return naive_ngram_rows(
        tokens, prev_tokens, MULTIPLIERS, HEAD_OFFSETS, HEAD_VOCAB,
        NGRAM, HEADS_PER_NGRAM, EOS,
    )


# ---------------------------------------------------------------------------
# ngram_rows
# ---------------------------------------------------------------------------


def test_ngram_rows_matches_naive_reference():
    rng = np.random.default_rng(0)
    for trial in range(20):
        B = int(rng.integers(1, 4))
        T = int(rng.integers(1, 17))
        tokens = rng.integers(0, VOCAB, size=(B, T)).astype(np.int64)
        # salt with plenty of EOS
        tokens[rng.random((B, T)) < 0.3] = EOS
        prev = rng.integers(0, VOCAB, size=(B, NGRAM - 1)).astype(np.int64)
        prev[rng.random((B, NGRAM - 1)) < 0.3] = EOS
        prev[rng.random((B, NGRAM - 1)) < 0.2] = -1  # missing predecessors

        got = call_rows(tokens, prev)
        want = naive_call(tokens, prev)
        assert got.dtype == np.int32
        assert got.shape == (B, T, N_HEADS)
        np.testing.assert_array_equal(got, want, err_msg=f"trial {trial}")


def test_ngram_rows_multiplier_actually_wraps():
    """Guard the whole point of uint64: a Python-int hash would differ here."""
    tokens = np.array([[VOCAB - 1]], dtype=np.int64)
    prev = np.array([[3, 5]], dtype=np.int64)
    got = call_rows(tokens, prev)

    MASK = 0xFFFFFFFFFFFFFFFF
    # prev is oldest-first, so ctx[1] == 5 and ctx[2] == 3
    c0, c1 = int(tokens[0, 0]), 5
    m0 = int(np.int64(MULTIPLIERS[0]).astype(np.uint64))
    m1 = int(np.int64(MULTIPLIERS[1]).astype(np.uint64))

    # head 0 is a bigram head: mixed = ctx0*m0 ^ ctx1*m1
    unbounded = (c0 * m0) ^ (c1 * m1)
    wrapped = ((c0 * m0) & MASK) ^ ((c1 * m1) & MASK)
    assert (c0 * m0) != ((c0 * m0) & MASK), "pick a multiplier that actually overflows"
    assert unbounded != wrapped, "the two hashes must genuinely disagree"

    v0, o0 = int(HEAD_VOCAB[0]), int(HEAD_OFFSETS[0])
    assert int(got[0, 0, 0]) == wrapped % v0 + o0
    assert int(got[0, 0, 0]) != unbounded % v0 + o0


def _rows_for(ctx0, ctx1, ctx2):
    """Rows a token would get for an explicit, already-resolved context."""
    with np.errstate(over="ignore"):
        m = MULTIPLIERS.astype(np.uint64)
        c = np.array([ctx0, ctx1, ctx2], dtype=np.int64).astype(np.uint64)
        bi = (c[0] * m[0]) ^ (c[1] * m[1])
        tri = bi ^ (c[2] * m[2])
    out = np.zeros(N_HEADS, dtype=np.int32)
    for g in range(HEADS_PER_NGRAM):
        out[g] = int(bi % np.uint64(HEAD_VOCAB[g])) + int(HEAD_OFFSETS[g])
        h = HEADS_PER_NGRAM + g
        out[h] = int(tri % np.uint64(HEAD_VOCAB[h])) + int(HEAD_OFFSETS[h])
    return out


def test_ngram_rows_eos_reset():
    """EOS 1 back and 2 back collapse the older slots; ctx[0]==EOS does not cut."""
    A, Bt, C = 11, 12, 13
    prev = np.array([[-1, -1]], dtype=np.int64)

    # EOS one position back: slot 1 AND the older slot 2 read as EOS
    rows = call_rows(np.array([[A, EOS, C]], dtype=np.int64), prev)
    np.testing.assert_array_equal(rows[0, 2], _rows_for(C, EOS, EOS))

    # EOS two positions back: slot 1 survives, slot 2 collapses
    rows = call_rows(np.array([[EOS, Bt, C]], dtype=np.int64), prev)
    np.testing.assert_array_equal(rows[0, 2], _rows_for(C, Bt, EOS))

    # no EOS anywhere: nothing collapses
    rows = call_rows(np.array([[A, Bt, C]], dtype=np.int64), prev)
    np.testing.assert_array_equal(rows[0, 2], _rows_for(C, Bt, A))

    # the token's OWN eos does not cut its own context
    rows = call_rows(np.array([[A, Bt, EOS]], dtype=np.int64), prev)
    np.testing.assert_array_equal(rows[0, 2], _rows_for(EOS, Bt, A))
    assert not np.array_equal(rows[0, 2], _rows_for(EOS, EOS, EOS))

    # missing predecessors at sequence start read as EOS
    rows = call_rows(np.array([[A, Bt, C]], dtype=np.int64), prev)
    np.testing.assert_array_equal(rows[0, 0], _rows_for(A, EOS, EOS))
    np.testing.assert_array_equal(rows[0, 1], _rows_for(Bt, A, EOS))

    # a supplied predecessor that is EOS cuts just like an in-window one
    rows = call_rows(np.array([[A]], dtype=np.int64),
                     np.array([[C, EOS]], dtype=np.int64))
    np.testing.assert_array_equal(rows[0, 0], _rows_for(A, EOS, EOS))
    rows = call_rows(np.array([[A]], dtype=np.int64),
                     np.array([[EOS, C]], dtype=np.int64))
    np.testing.assert_array_equal(rows[0, 0], _rows_for(A, C, EOS))


def _chunk_rows(seq, chunk_sizes):
    """Rows for a [B,T] sequence, computed chunk by chunk with threaded prev_tokens."""
    B, T = seq.shape
    n_prev = NGRAM - 1
    parts = []
    pos = 0
    for size in chunk_sizes:
        prev = np.full((B, n_prev), -1, dtype=np.int64)
        take = min(n_prev, pos)
        if take:
            prev[:, n_prev - take:] = seq[:, pos - take: pos]
        parts.append(call_rows(seq[:, pos: pos + size], prev))
        pos += size
    assert pos == T, f"chunk sizes {chunk_sizes} do not cover T={T}"
    return np.concatenate(parts, axis=1)


def test_ngram_rows_chunked_equals_single_shot():
    rng = np.random.default_rng(1)
    for _ in range(5):
        seq = rng.integers(0, VOCAB, size=(2, 12)).astype(np.int64)
        seq[rng.random((2, 12)) < 0.25] = EOS
        one_shot = call_rows(seq, np.full((2, NGRAM - 1), -1, dtype=np.int64))
        np.testing.assert_array_equal(_chunk_rows(seq, [1, 5, 6]), one_shot)
        np.testing.assert_array_equal(_chunk_rows(seq, [1] * 12), one_shot)
        np.testing.assert_array_equal(_chunk_rows(seq, [6, 6]), one_shot)
        np.testing.assert_array_equal(_chunk_rows(seq, [12]), one_shot)


def test_ngram_rows_in_bounds():
    rng = np.random.default_rng(2)
    seq = rng.integers(0, VOCAB, size=(3, 64)).astype(np.int64)
    seq[rng.random((3, 64)) < 0.3] = EOS
    prev = rng.integers(-1, VOCAB, size=(3, NGRAM - 1)).astype(np.int64)
    rows = call_rows(seq, prev)
    assert rows.min() >= 0, f"negative row index {rows.min()}"
    assert rows.max() < N_ROWS, f"row {rows.max()} >= n_rows {N_ROWS}"
    # each head stays inside its own [offset, offset+vocab) slice
    for h in range(N_HEADS):
        lo, hi = int(HEAD_OFFSETS[h]), int(HEAD_OFFSETS[h] + HEAD_VOCAB[h])
        assert rows[:, :, h].min() >= lo
        assert rows[:, :, h].max() < hi


def test_ngram_rows_rejects_bad_shapes():
    toks = np.zeros((2, 5), dtype=np.int64)
    with pytest.raises(ValueError):
        call_rows(toks, np.zeros((2, 1), dtype=np.int64))       # wrong n_prev
    with pytest.raises(ValueError):
        call_rows(np.zeros(5, dtype=np.int64), np.zeros((1, 2), dtype=np.int64))
    with pytest.raises(ValueError):                            # offsets too short
        ngram_rows(toks, np.zeros((2, 2), dtype=np.int64), MULTIPLIERS,
                   HEAD_OFFSETS[:2], HEAD_VOCAB, NGRAM, HEADS_PER_NGRAM, EOS)
    with pytest.raises(ValueError):                            # ngram_size < 2
        ngram_rows(toks, np.zeros((2, 0), dtype=np.int64), MULTIPLIERS,
                   HEAD_OFFSETS, HEAD_VOCAB, 1, HEADS_PER_NGRAM, EOS)


# ---------------------------------------------------------------------------
# grouped_norm
# ---------------------------------------------------------------------------


def test_grouped_norm_matches_reference_loop():
    args = toy_args()
    B, T, hc, D = 2, 3, args.hc_count, args.hidden_size
    mx.random.seed(0)
    x = mx.random.normal((B, T, hc * D))
    w = mx.random.normal((hc * D,))
    got = np.array(grouped_norm(x, w, hc, args.rms_norm_eps))

    xn = np.array(x).reshape(B, T, hc, D).astype(np.float64)
    ref = np.empty_like(xn)
    for b in range(B):
        for t in range(T):
            for c in range(hc):
                v = xn[b, t, c]
                ref[b, t, c] = v / np.sqrt((v * v).mean() + args.rms_norm_eps)
    ref = ref.reshape(B, T, hc * D) * np.array(w)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# PLEBlock
# ---------------------------------------------------------------------------


def build_block(seed=0):
    mx.random.seed(seed)
    args = toy_args()
    block = PLEBlock(args)
    # non-trivial norms and conv taps so no bug can hide behind a zero
    block.norm_key = mx.random.normal((args.hc_dim,)) * 0.2 + 1.0
    block.norm_query = mx.random.normal((args.hc_dim,)) * 0.2 + 1.0
    block.norm_conv = mx.random.normal((args.hc_dim,)) * 0.2 + 1.0
    block.conv1d = mx.random.normal((args.hc_dim, args.ple_conv_kernel_size)) * 0.5
    table = mx.random.normal((N_ROWS, args.ple_head_dim))
    return args, block, table


def test_ple_block_shapes():
    args, block, table = build_block()
    B, T = 2, 5
    hidden = mx.random.normal((B, T, args.hc_count, args.hidden_size))
    rng = np.random.default_rng(3)
    seq = rng.integers(0, VOCAB, size=(B, T)).astype(np.int64)
    rows = call_rows(seq, np.full((B, NGRAM - 1), -1, dtype=np.int64))

    out, state = block(hidden, rows, table)
    assert out.shape == (B, T, args.hc_count, args.hidden_size)
    assert state.shape == (B, args.ple_conv_history, args.hc_dim)
    assert args.ple_conv_history == 9
    assert args.ple_n_heads == N_HEADS
    assert args.ple_head_dim == 2
    # accepts an mx.array of rows just as happily as a numpy one
    out2, _ = block(hidden, mx.array(rows), table)
    np.testing.assert_array_equal(np.array(out), np.array(out2))


def test_ple_block_parameter_names():
    args, block, table = build_block()
    params = block.parameters()
    assert set(params) == {
        "key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d",
    }
    assert params["key_proj"]["weight"].shape == (args.hc_dim, args.ple_embed_dim)
    assert params["value_proj"]["weight"].shape == (args.hidden_size, args.ple_embed_dim)
    assert "bias" not in params["key_proj"]
    assert "bias" not in params["value_proj"]
    assert params["norm_key"].shape == (args.hc_dim,)
    assert params["norm_query"].shape == (args.hc_dim,)
    assert params["norm_conv"].shape == (args.hc_dim,)
    assert params["conv1d"].shape == (args.hc_dim, args.ple_conv_kernel_size)


def test_ple_block_no_nan():
    args, block, table = build_block(seed=5)
    B, T = 2, 12
    hidden = mx.random.normal((B, T, args.hc_count, args.hidden_size)) * 10.0
    rng = np.random.default_rng(4)
    seq = rng.integers(0, VOCAB, size=(B, T)).astype(np.int64)
    seq[rng.random((B, T)) < 0.3] = EOS
    rows = call_rows(seq, np.full((B, NGRAM - 1), -1, dtype=np.int64))
    out, state = block(hidden, rows, table)
    assert np.isfinite(np.array(out)).all(), "out has NaN/Inf"
    assert np.isfinite(np.array(state)).all(), "state has NaN/Inf"

    # a zeroed hidden stream drives s to exactly 0 -> exercises the sign(0) branch
    out0, _ = block(mx.zeros_like(hidden), rows, table)
    assert np.isfinite(np.array(out0)).all()


def test_ple_block_dilated_conv_indexing():
    """Isolate the conv: tap k must read (K-1-k)*dilation positions back from t,
    drawing on the carried history when that runs off the front of the chunk."""
    args, block, table = build_block(seed=6)
    B, T = 1, 12
    K, dil = args.ple_conv_kernel_size, args.ngram_size

    hidden = mx.random.normal((B, T, args.hc_count, args.hidden_size))
    rng = np.random.default_rng(7)
    seq = rng.integers(0, VOCAB, size=(B, T)).astype(np.int64)
    rows = call_rows(seq, np.full((B, NGRAM - 1), -1, dtype=np.int64))

    state = mx.random.normal((B, args.ple_conv_history, args.hc_dim))
    out, new_state = block(hidden, rows, table, state)

    # recompute every stage independently from the pieces the block exposes
    emb = table[mx.array(rows)].reshape(B, T, args.ple_embed_dim)
    key = grouped_norm(block.key_proj(emb), block.norm_key, args.hc_count,
                       args.rms_norm_eps).reshape(B, T, args.hc_count, args.hidden_size)
    query = grouped_norm(hidden.reshape(B, T, args.hc_dim), block.norm_query,
                         args.hc_count, args.rms_norm_eps).reshape(
                             B, T, args.hc_count, args.hidden_size)
    s = (key * query).sum(-1) / (args.hidden_size ** 0.5)
    gate = mx.sigmoid(mx.sign(s) * mx.sqrt(mx.maximum(mx.abs(s), 1e-6)))
    value = block.value_proj(emb)
    gated = value[:, :, None, :] * gate[:, :, :, None]
    norm = np.array(grouped_norm(gated.reshape(B, T, args.hc_dim), block.norm_conv,
                                 args.hc_count, args.rms_norm_eps)).astype(np.float64)

    hist = np.array(state).astype(np.float64)
    w = np.array(block.conv1d).astype(np.float64)
    ref = np.zeros((B, T, args.hc_dim), dtype=np.float64)
    for t in range(T):
        for k in range(K):
            src = t - (K - 1 - k) * dil        # positions back from t
            x = norm[:, src, :] if src >= 0 else hist[:, hist.shape[1] + src, :]
            ref[:, t, :] += w[:, k] * x
    ref = ref / (1.0 + np.exp(-ref))           # silu
    expect = np.array(hidden) + np.array(gated) + ref.reshape(
        B, T, args.hc_count, args.hidden_size)
    np.testing.assert_allclose(np.array(out), expect, rtol=1e-4, atol=1e-4)

    # the returned state is the last `history` rows of [state | normalized]
    padded = np.concatenate([hist, norm], axis=1)
    np.testing.assert_allclose(
        np.array(new_state), padded[:, -args.ple_conv_history:, :],
        rtol=1e-5, atol=1e-5,
    )


def test_ple_block_conv_taps_are_distinct():
    """A one-hot conv kernel on tap k must reproduce a pure shift by (K-1-k)*dilation.
    If the taps were reversed or the dilation dropped this would catch it."""
    args, block, table = build_block(seed=12)
    B, T = 1, 12
    K, dil = args.ple_conv_kernel_size, args.ngram_size
    block.norm_key = mx.ones((args.hc_dim,))
    block.norm_query = mx.ones((args.hc_dim,))
    block.norm_conv = mx.ones((args.hc_dim,))

    hidden = mx.random.normal((B, T, args.hc_count, args.hidden_size))
    rng = np.random.default_rng(13)
    seq = rng.integers(0, VOCAB, size=(B, T)).astype(np.int64)
    rows = call_rows(seq, np.full((B, NGRAM - 1), -1, dtype=np.int64))
    state = mx.random.normal((B, args.ple_conv_history, args.hc_dim))

    # dense reference for `normalized`, taken from the K=all-zeros run
    block.conv1d = mx.zeros((args.hc_dim, K))
    base, _ = block(hidden, rows, table, state)
    base = np.array(base)  # hidden + gated + silu(0) == hidden + gated

    hist = np.array(state).astype(np.float64)

    def tap_out(k):
        onehot = np.zeros((args.hc_dim, K), dtype=np.float32)
        onehot[:, k] = 1.0
        block.conv1d = mx.array(onehot)
        out = np.array(block(hidden, rows, table, state)[0]).astype(np.float64)
        return (out - base).reshape(B, T, args.hc_dim)

    # tap K-1 has zero lag, so it IS silu(normalized[t]); every other tap must be a
    # pure shift of it (or of the carried history) by (K-1-k)*dilation.
    identity_pre = tap_out(K - 1)

    for k in range(K - 1):
        conv = tap_out(k)
        back = (K - 1 - k) * dil
        for t in range(T):
            src = t - back
            if src >= 0:
                np.testing.assert_allclose(
                    conv[:, t, :], identity_pre[:, src, :], rtol=1e-4, atol=1e-4,
                    err_msg=f"tap {k} at t={t} does not read {back} back",
                )
            else:
                h = hist[:, hist.shape[1] + src, :]
                np.testing.assert_allclose(
                    conv[:, t, :], h / (1.0 + np.exp(-h)), rtol=1e-4, atol=1e-4,
                    err_msg=f"tap {k} at t={t} does not read history slot {src}",
                )


def test_ple_block_chunked_prefill_equivalence():
    args, block, table = build_block(seed=8)
    B, T = 2, 12
    mx.random.seed(11)
    hidden = mx.random.normal((B, T, args.hc_count, args.hidden_size))
    rng = np.random.default_rng(9)
    seq = rng.integers(0, VOCAB, size=(B, T)).astype(np.int64)
    seq[rng.random((B, T)) < 0.25] = EOS

    rows_full = call_rows(seq, np.full((B, NGRAM - 1), -1, dtype=np.int64))
    out_full, state_full = block(hidden, rows_full, table)
    out_full = np.array(out_full)
    state_full = np.array(state_full)
    n_prev = NGRAM - 1

    for chunks in ([1, 5, 6], [1] * 12, [6, 6], [5, 1, 6], [12]):
        pos = 0
        state = None
        pieces = []
        for size in chunks:
            prev = np.full((B, n_prev), -1, dtype=np.int64)
            take = min(n_prev, pos)
            if take:
                prev[:, n_prev - take:] = seq[:, pos - take: pos]
            rows = call_rows(seq[:, pos: pos + size], prev)
            piece, state = block(hidden[:, pos: pos + size], rows, table, state)
            pieces.append(piece)
            pos += size
        assert pos == T
        got = np.array(mx.concatenate(pieces, axis=1))
        np.testing.assert_allclose(
            got, out_full, rtol=1e-4, atol=1e-5, err_msg=f"chunks={chunks}",
        )
        np.testing.assert_allclose(
            np.array(state), state_full, rtol=1e-4, atol=1e-5,
            err_msg=f"final conv state, chunks={chunks}",
        )


def test_ple_block_rejects_bad_shapes():
    args, block, table = build_block()
    B, T = 1, 3
    hidden = mx.zeros((B, T, args.hc_count, args.hidden_size))
    rows = np.zeros((B, T, N_HEADS), dtype=np.int32)

    with pytest.raises(ValueError):
        block(mx.zeros((B, T, args.hc_dim)), rows, table)          # not wide
    with pytest.raises(ValueError):
        block(mx.zeros((B, T, args.hc_count + 1, args.hidden_size)), rows, table)
    with pytest.raises(ValueError):
        block(hidden, np.zeros((B, T, N_HEADS + 1), dtype=np.int32), table)
    with pytest.raises(ValueError):
        block(hidden, rows, mx.zeros((N_ROWS, args.ple_head_dim + 1)))
    with pytest.raises(ValueError):
        block(hidden, rows, table,
              mx.zeros((B, args.ple_conv_history + 1, args.hc_dim)))
