"""Tests for mlx_qwen4exp.hyper — the hyper-connection (gated residual) machinery.

Toy dimensions throughout: hidden_size=8, hc_count=4, hc_lowrank=6 => hc_dim=32.
No checkpoint required; all weights are random.

Note on laziness: MLX shapes are known without executing the graph, so shape assertions
need no forcing. Every value assertion here goes through ``np.array(...)`` or a
bool-conversion of an ``mx`` predicate, both of which force the computation -- so a NaN
produced anywhere in the graph will surface.

Run:
    /opt/homebrew/bin/python3.11 -m pytest tests/test_hyper.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# repo root on sys.path so `mlx_qwen4exp` (an implicit namespace package) imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mlx_qwen4exp.config import ModelArgs  # noqa: E402
from mlx_qwen4exp.hyper import (  # noqa: E402
    HyperConnection,
    grouped_rms_norm,
    hc_combine,
    hc_init,
)

HIDDEN = 8
HC = 4
LOWRANK = 6
HC_DIM = HC * HIDDEN  # 32
EPS = 1e-6


def toy_args(**overrides) -> ModelArgs:
    kwargs = dict(
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        num_hidden_layers=4,
        full_attention_interval=4,
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


def randomize(hc_module: HyperConnection, seed: int = 0) -> HyperConnection:
    """Give the module non-trivial weights so nothing is accidentally an identity."""
    mx.random.seed(seed)
    hc_module.hc_norm = mx.random.normal((HC_DIM,)) * 0.5 + 1.0
    hc_module.input_mix_weight_down.weight = mx.random.normal((LOWRANK, HC_DIM)) * 0.1
    hc_module.input_mix_weight_up.weight = mx.random.normal((HC_DIM, LOWRANK)) * 0.1
    if hc_module.with_inject:
        hc_module.block_inject_weight.weight = mx.random.normal((HC, HC_DIM)) * 0.1
    return hc_module


def silu(z: mx.array) -> mx.array:
    return z * mx.sigmoid(z)


# ---------------------------------------------------------------- config sanity
def test_toy_args_derived_dims():
    args = toy_args()
    assert args.hc_dim == HC_DIM
    assert args.hc_count == HC
    assert args.hc_lowrank == LOWRANK


# ---------------------------------------------------------------- shapes
def test_hyperconnection_shapes_with_inject():
    args = toy_args()
    hcm = randomize(HyperConnection(args, with_inject=True))
    b, t = 2, 5
    x = mx.random.normal((b, t, HC, HIDDEN))
    mixed, inject = hcm(x)
    assert mixed.shape == (b, t, HIDDEN)
    assert inject is not None
    assert inject.shape == (b, t, HC)
    assert np.isfinite(np.array(mixed)).all()
    assert np.isfinite(np.array(inject)).all()


def test_hyperconnection_shapes_without_inject():
    args = toy_args()
    hcm = randomize(HyperConnection(args, with_inject=False))
    b, t = 3, 1
    x = mx.random.normal((b, t, HC, HIDDEN))
    mixed, inject = hcm(x)
    assert mixed.shape == (b, t, HIDDEN)
    assert inject is None
    assert not hasattr(hcm, "block_inject_weight")
    assert np.isfinite(np.array(mixed)).all()


def test_hyperconnection_parameter_names_and_shapes():
    """Weight loading depends on these exact attribute names / shapes."""
    args = toy_args()
    hcm = HyperConnection(args, with_inject=True)
    params = dict(tree_flatten(hcm.parameters()))
    assert set(params) == {
        "hc_norm",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    }
    assert params["hc_norm"].shape == (HC_DIM,)
    assert params["input_mix_weight_down.weight"].shape == (LOWRANK, HC_DIM)
    assert params["input_mix_weight_up.weight"].shape == (HC_DIM, LOWRANK)
    assert params["block_inject_weight.weight"].shape == (HC, HC_DIM)

    noinj = dict(tree_flatten(HyperConnection(args, with_inject=False).parameters()))
    assert "block_inject_weight.weight" not in noinj


def test_real_config_parameter_shapes():
    """Default (production) config must produce the checkpoint's shapes exactly."""
    args = ModelArgs()
    hcm = HyperConnection(args, with_inject=True)
    params = dict(tree_flatten(hcm.parameters()))
    assert params["hc_norm"].shape == (10240,)
    assert params["input_mix_weight_down.weight"].shape == (320, 10240)
    assert params["input_mix_weight_up.weight"].shape == (10240, 320)
    assert params["block_inject_weight.weight"].shape == (4, 10240)


def test_hyperconnection_rejects_wrong_rank_and_dims():
    args = toy_args()
    hcm = randomize(HyperConnection(args))
    with pytest.raises(ValueError):
        hcm(mx.zeros((2, 5, HC_DIM)))  # already flattened
    with pytest.raises(ValueError):
        hcm(mx.zeros((2, 5, HC + 1, HIDDEN)))  # wrong stream count
    with pytest.raises(ValueError):
        hcm(mx.zeros((2, 5, HC, HIDDEN + 1)))  # wrong hidden size


# ---------------------------------------------------------------- hc_init
def test_hc_init_streams_identical_and_equal_input():
    b, t = 2, 6
    emb = mx.random.normal((b, t, HIDDEN))
    wide = hc_init(emb, HC)
    assert wide.shape == (b, t, HC, HIDDEN)
    for c in range(HC):
        assert bool(mx.array_equal(wide[:, :, c, :], emb)), f"stream {c} != input"
    for c in range(1, HC):
        assert bool(mx.array_equal(wide[:, :, c, :], wide[:, :, 0, :]))


def test_hc_init_result_is_materialised_and_per_stream_independent():
    """Per-stream updates must not alias through a zero-stride broadcast view.

    HONEST NOTE: this test does NOT discriminate. MLX arrays have value semantics --
    ``a[i] = v`` lowers to ``slice_update`` and returns a new array -- so a broadcast
    view cannot alias here, and this passes with or without the ``mx.contiguous`` in
    ``hc_init`` (verified by mutation). It is kept as a behavioural invariant guard for
    the day this code is ported to a framework where the distinction bites.
    """
    emb = mx.random.normal((1, 3, HIDDEN))
    wide = hc_init(emb, HC)
    bump = mx.zeros((1, 3, HC, HIDDEN))
    bump[:, :, 1, :] = mx.ones((1, 3, HIDDEN))
    out = wide + bump
    assert bool(mx.allclose(out[:, :, 1, :], emb + 1.0))
    assert bool(mx.allclose(out[:, :, 0, :], emb))
    assert bool(mx.allclose(out[:, :, 2, :], emb))
    assert bool(mx.allclose(out[:, :, 3, :], emb))

    # in-place style write into one stream must not touch the others or the source
    wide[:, :, 0, :] = mx.zeros((1, 3, HIDDEN))
    assert np.allclose(np.array(wide[:, :, 0, :]), 0.0)
    assert bool(mx.allclose(wide[:, :, 1, :], emb)), "stream 1 aliased stream 0"
    assert bool(mx.allclose(wide[:, :, 2, :], emb))
    assert bool(mx.allclose(wide[:, :, 3, :], emb))
    assert not np.allclose(np.array(emb), 0.0), "source embeddings were clobbered"


def test_hc_init_rejects_bad_input():
    with pytest.raises(ValueError):
        hc_init(mx.zeros((2, HIDDEN)), HC)
    with pytest.raises(ValueError):
        hc_init(mx.zeros((1, 2, HIDDEN)), 0)


# ---------------------------------------------------------------- hc_combine
def test_hc_combine_zero_inject_is_plain_residual_add():
    """inject == 0 => w == 2*sigmoid(0) == 1 exactly, in every stream."""
    b, t = 2, 7
    res = mx.random.normal((b, t, HC, HIDDEN))
    block = mx.random.normal((b, t, HIDDEN))
    inject = mx.zeros((b, t, HC))
    out = hc_combine(res, block, inject, HC)
    expected = res + mx.broadcast_to(block[:, :, None, :], res.shape)
    assert bool(mx.allclose(out, expected, rtol=0, atol=0)), "zero injection must be exact"
    for c in range(HC):
        assert bool(mx.allclose(out[:, :, c, :], res[:, :, c, :] + block))


def test_hc_combine_zero_block_is_identity():
    b, t = 1, 4
    res = mx.random.normal((b, t, HC, HIDDEN))
    out = hc_combine(res, mx.zeros((b, t, HIDDEN)), mx.random.normal((b, t, HC)), HC)
    assert bool(mx.allclose(out, res, rtol=0, atol=0))


def test_hc_combine_matches_numpy_reference():
    b, t = 2, 3
    mx.random.seed(11)
    res = mx.random.normal((b, t, HC, HIDDEN))
    block = mx.random.normal((b, t, HIDDEN))
    inject = mx.random.normal((b, t, HC)) * 3.0
    out = np.array(hc_combine(res, block, inject, HC))

    r = np.array(res, dtype=np.float64)
    bl = np.array(block, dtype=np.float64)
    inj = np.array(inject, dtype=np.float64)
    w = 2.0 / (1.0 + np.exp(-inj / HC))
    ref = r.copy()
    for i in range(b):
        for j in range(t):
            for c in range(HC):
                ref[i, j, c, :] = r[i, j, c, :] + bl[i, j, :] * w[i, j, c]
    assert np.allclose(out, ref, atol=1e-5)


def test_hc_combine_shape_validation():
    b, t = 2, 3
    res = mx.zeros((b, t, HC, HIDDEN))
    block = mx.zeros((b, t, HIDDEN))
    inject = mx.zeros((b, t, HC))
    with pytest.raises(ValueError):
        hc_combine(mx.zeros((b, t, HIDDEN)), block, inject, HC)  # residual not wide
    with pytest.raises(ValueError):
        hc_combine(res, mx.zeros((b, t, HC, HIDDEN)), inject, HC)  # block wide
    with pytest.raises(ValueError):
        hc_combine(res, block, mx.zeros((b, t, HC + 1)), HC)  # inject wrong hc
    with pytest.raises(ValueError):
        hc_combine(res, mx.zeros((b, t, HIDDEN + 1)), inject, HC)  # block wrong D
    with pytest.raises(ValueError):
        hc_combine(res, block, inject, HC + 1)  # hc_count mismatch


# ---------------------------------------------------------------- grouped norm
def reference_grouped_norm(
    x_wide_np: np.ndarray, gamma_np: np.ndarray, eps: float
) -> np.ndarray:
    """INDEPENDENT reference: explicit python loop over the hc streams.

    Normalise each [.., D] slice separately (reduce over D only), flatten the streams
    into hc*D in stream-major order, then apply the flattened gamma.
    """
    b, t, hc, d = x_wide_np.shape
    out = np.zeros((b, t, hc * d), dtype=np.float64)
    for i in range(b):
        for j in range(t):
            for c in range(hc):
                v = x_wide_np[i, j, c, :].astype(np.float64)
                ms = 0.0
                for k in range(d):
                    ms += v[k] * v[k]
                ms /= d
                scale = 1.0 / math.sqrt(ms + eps)
                for k in range(d):
                    out[i, j, c * d + k] = v[k] * scale
    return out * gamma_np.astype(np.float64)


def test_grouped_norm_matches_loop_reference():
    mx.random.seed(3)
    b, t = 2, 4
    x = mx.random.normal((b, t, HC, HIDDEN)) * 2.0
    gamma = mx.random.normal((HC_DIM,)) * 0.5 + 1.0
    got = np.array(grouped_rms_norm(x, gamma, EPS), dtype=np.float64)
    ref = reference_grouped_norm(np.array(x), np.array(gamma), EPS)
    assert got.shape == ref.shape == (b, t, HC_DIM)
    assert np.allclose(got, ref, atol=1e-5), np.abs(got - ref).max()


def test_grouped_norm_differs_from_plain_flat_norm():
    """Guards against accidentally normalising over hc*D instead of per-stream D."""
    mx.random.seed(4)
    x = mx.random.normal((1, 2, HC, HIDDEN))
    # make stream magnitudes wildly different so the two norms cannot coincide
    scales = mx.array([1.0, 10.0, 100.0, 1000.0]).reshape(1, 1, HC, 1)
    x = x * scales
    grouped = np.array(grouped_rms_norm(x, None, EPS), dtype=np.float64)
    flat = np.array(x).reshape(1, 2, HC_DIM).astype(np.float64)
    plain = flat / np.sqrt((flat**2).mean(axis=-1, keepdims=True) + EPS)
    assert not np.allclose(grouped, plain, atol=1e-3)
    # per-stream RMS of the grouped result must be ~1 for every stream
    per_stream = grouped.reshape(1, 2, HC, HIDDEN)
    rms = np.sqrt((per_stream**2).mean(axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-4), rms


def test_grouped_norm_none_gamma_and_bad_gamma():
    x = mx.random.normal((1, 2, HC, HIDDEN))
    assert grouped_rms_norm(x, None, EPS).shape == (1, 2, HC_DIM)
    with pytest.raises(ValueError):
        grouped_rms_norm(x, mx.ones((HC_DIM + 1,)), EPS)
    with pytest.raises(ValueError):
        grouped_rms_norm(mx.zeros((HIDDEN,)), None, EPS)


# ---------------------------------------------------------------- full forward math
def reference_hc_mix(x_np, gamma, w_down, w_up, w_inject, hc, d, eps):
    """Straight NumPy re-derivation of SPEC 4.1 / qwen4exp.cpp build_hc_mix."""
    xn = reference_grouped_norm(x_np, gamma, eps)  # [B, T, hc*D]
    lo = xn @ w_down.astype(np.float64).T          # [B, T, lowrank]
    lo = lo / hc                                   # ggml_scale BEFORE the silu
    lo = lo / (1.0 + np.exp(-lo))                  # silu
    gate = 1.0 / (1.0 + np.exp(-(lo @ w_up.astype(np.float64).T)))
    gated = (xn * gate).reshape(*x_np.shape[:2], hc, d)
    mixed = gated.mean(axis=2)
    inject = xn @ w_inject.astype(np.float64).T
    return mixed, inject


def test_hyperconnection_matches_numpy_reference():
    args = toy_args()
    hcm = randomize(HyperConnection(args), seed=7)
    b, t = 2, 3
    mx.random.seed(21)
    x = mx.random.normal((b, t, HC, HIDDEN)) * 1.5
    mixed, inject = hcm(x)

    ref_mixed, ref_inject = reference_hc_mix(
        np.array(x),
        np.array(hcm.hc_norm),
        np.array(hcm.input_mix_weight_down.weight),
        np.array(hcm.input_mix_weight_up.weight),
        np.array(hcm.block_inject_weight.weight),
        HC,
        HIDDEN,
        EPS,
    )
    assert np.allclose(np.array(mixed), ref_mixed, atol=1e-5), np.abs(
        np.array(mixed) - ref_mixed
    ).max()
    assert np.allclose(np.array(inject), ref_inject, atol=1e-5)


def test_scale_by_hc_is_before_silu_not_after():
    """Regression guard: silu(z/hc) != silu(z)/hc. Catches a misplaced ggml_scale."""
    args = toy_args()
    hcm = randomize(HyperConnection(args), seed=9)
    # big down-projection weights so the two orderings diverge sharply
    hcm.input_mix_weight_down.weight = mx.random.normal((LOWRANK, HC_DIM)) * 2.0
    x = mx.random.normal((1, 2, HC, HIDDEN))
    mixed, _ = hcm(x)

    xn = grouped_rms_norm(x, hcm.hc_norm, EPS)
    z = hcm.input_mix_weight_down(xn)
    wrong_lo = silu(z) / HC
    wrong_gate = mx.sigmoid(hcm.input_mix_weight_up(wrong_lo))
    wrong = (xn * wrong_gate).reshape(1, 2, HC, HIDDEN).mean(axis=2)
    assert not bool(mx.allclose(mixed, wrong, atol=1e-4))


# ---------------------------------------------------------------- numerics
@pytest.mark.parametrize("magnitude", [1e4, 1e-8, 1e8, 1e-20, 0.0])
def test_no_nan_with_extreme_inputs(magnitude):
    args = toy_args()
    hcm = randomize(HyperConnection(args), seed=13)
    mx.random.seed(31)
    x = mx.random.normal((2, 3, HC, HIDDEN)) * magnitude
    mixed, inject = hcm(x)
    for name, arr in (("mixed", mixed), ("inject", inject)):
        a = np.array(arr)
        assert not np.isnan(a).any(), f"{name} has NaN at magnitude {magnitude}"
        assert np.isfinite(a).all(), f"{name} has Inf at magnitude {magnitude}"


def test_no_nan_all_zero_stream():
    """An exactly-zero residual stream must not divide by zero (eps saves it)."""
    args = toy_args()
    hcm = randomize(HyperConnection(args), seed=17)
    x = mx.zeros((1, 2, HC, HIDDEN))
    mixed, inject = hcm(x)
    assert np.isfinite(np.array(mixed)).all()
    assert np.isfinite(np.array(inject)).all()
    # zero in => xn is exactly 0 => gated is 0 => mixed and inject are 0
    assert np.allclose(np.array(mixed), 0.0, atol=1e-6)
    assert np.allclose(np.array(inject), 0.0, atol=1e-6)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_dtype_is_preserved_and_finite(dtype):
    """Production runs bf16/fp16; the norm must not silently upcast the output."""
    args = toy_args()
    hcm = randomize(HyperConnection(args), seed=23)
    hcm.hc_norm = hcm.hc_norm.astype(dtype)
    hcm.input_mix_weight_down.weight = hcm.input_mix_weight_down.weight.astype(dtype)
    hcm.input_mix_weight_up.weight = hcm.input_mix_weight_up.weight.astype(dtype)
    hcm.block_inject_weight.weight = hcm.block_inject_weight.weight.astype(dtype)

    x = (mx.random.normal((2, 3, HC, HIDDEN)) * 2.0).astype(dtype)
    mixed, inject = hcm(x)
    assert mixed.dtype == dtype
    assert inject.dtype == dtype
    assert np.isfinite(np.array(mixed.astype(mx.float32))).all()
    assert np.isfinite(np.array(inject.astype(mx.float32))).all()

    res = hc_init(x[:, :, 0, :], HC)
    out = hc_combine(res, mixed, inject, HC)
    assert out.dtype == dtype
    assert np.isfinite(np.array(out.astype(mx.float32))).all()


def test_hc_combine_no_nan_with_extreme_inject():
    b, t = 1, 3
    res = mx.random.normal((b, t, HC, HIDDEN)) * 1e4
    block = mx.random.normal((b, t, HIDDEN)) * 1e-8
    for mag in (1e8, -1e8, 1e-20):
        inject = mx.full((b, t, HC), mag)
        out = hc_combine(res, block, inject, HC)
        assert np.isfinite(np.array(out)).all(), f"non-finite at inject={mag}"


# ---------------------------------------------------------------- round trip
def test_block_site_round_trip_shapes_and_finiteness():
    """embed -> hc_init -> mix -> (identity block) -> combine, like a real layer stack."""
    args = toy_args()
    attn_hc = randomize(HyperConnection(args), seed=1)
    mlp_hc = randomize(HyperConnection(args), seed=2)
    final = randomize(HyperConnection(args, with_inject=False), seed=3)

    b, t = 2, 5
    emb = mx.random.normal((b, t, HIDDEN))
    res = hc_init(emb, HC)
    for _ in range(3):
        cur, inj = attn_hc(res)
        res = hc_combine(res, cur, inj, HC)
        cur, inj = mlp_hc(res)
        res = hc_combine(res, cur, inj, HC)
    out, inject = final(res)
    assert res.shape == (b, t, HC, HIDDEN)
    assert out.shape == (b, t, HIDDEN)
    assert inject is None
    assert np.isfinite(np.array(res)).all()
    assert np.isfinite(np.array(out)).all()


# ---------------------------------------------------------------- weight loading
def test_sanitize_prefix_renames_hc_norm_and_module_accepts_it():
    args = toy_args()
    prefix = "layers.0.attn_hyper_connection"
    ckpt = {
        f"{prefix}.hc_norm.weight": mx.random.normal((HC_DIM,)),
        f"{prefix}.input_mix_weight_down.weight": mx.random.normal((LOWRANK, HC_DIM)),
        f"{prefix}.input_mix_weight_up.weight": mx.random.normal((HC_DIM, LOWRANK)),
        f"{prefix}.block_inject_weight.weight": mx.random.normal((HC, HC_DIM)),
    }
    fixed = HyperConnection.sanitize_prefix(ckpt, prefix)
    assert f"{prefix}.hc_norm" in fixed
    assert f"{prefix}.hc_norm.weight" not in fixed
    assert f"{prefix}.hc_norm.weight" in ckpt, "input dict must not be mutated"

    hcm = HyperConnection(args, with_inject=True)
    stripped = [(k[len(prefix) + 1 :], v) for k, v in fixed.items()]
    hcm.update(tree_unflatten(stripped))
    assert bool(mx.array_equal(hcm.hc_norm, ckpt[f"{prefix}.hc_norm.weight"]))
    assert bool(
        mx.array_equal(
            hcm.block_inject_weight.weight, ckpt[f"{prefix}.block_inject_weight.weight"]
        )
    )
    mixed, inject = hcm(mx.random.normal((1, 2, HC, HIDDEN)))
    assert mixed.shape == (1, 2, HIDDEN)
    assert np.isfinite(np.array(mixed)).all()


def test_sanitize_prefix_is_a_noop_when_key_absent():
    d = {"foo.bar": mx.zeros((1,))}
    out = HyperConnection.sanitize_prefix(d, "layers.0.mlp_hyper_connection")
    assert set(out) == set(d)
    assert out is not d
