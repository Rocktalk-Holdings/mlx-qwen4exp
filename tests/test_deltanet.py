"""Tests for mlx_qwen4exp.deltanet -- the Gated DeltaNet (linear-attention) layer.

Toy dimensions where the fused Metal kernel permits, otherwise small real-ish dims.
No checkpoint required; all weights are random. float32 throughout for tight tolerances.

Forcing rule (harness quirk: no mx.eval in tests): value assertions go through
``np.array(...)`` / ``float(...)``, which force the graph, so any NaN surfaces.

Run:
    /opt/homebrew/bin/python3.11 -m pytest tests/test_deltanet.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# repo root on sys.path so `mlx_qwen4exp` imports as a namespace package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mlx_lm.models.cache import ArraysCache  # noqa: E402

from mlx_qwen4exp.config import ModelArgs  # noqa: E402
from mlx_qwen4exp.deltanet import (  # noqa: E402
    GatedDeltaNet,
    Qwen4ExpRMSNormGated,
    _l2_normalize,
)


# --------------------------------------------------------------------------- helpers

def toy_args(**overrides) -> ModelArgs:
    """Small dims that satisfy the delta-rule Metal kernel constraint (Dk must be a
    multiple of 32; the kernel computes n_per_t = Dk/32): head_dim 32.
    hidden=128, 4 key heads, 12 value heads (12 % 4 == 0), head_dim 32.
    conv_dim = 2*128 + 384 = 640.  value_dim = 12*32 = 384."""
    base = dict(
        hidden_size=128,
        num_hidden_layers=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_num_value_heads=12,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    base.update(overrides)
    return ModelArgs(**base)


def make_layer(args: ModelArgs, seed: int = 0) -> GatedDeltaNet:
    mx.random.seed(seed)
    layer = GatedDeltaNet(args)
    # randomise A_log / dt_bias so the decay path is exercised (not identity)
    layer.A_log = mx.random.uniform(low=-2.0, high=2.0, shape=(args.linear_num_value_heads,))
    layer.dt_bias = mx.random.uniform(low=-1.0, high=1.0, shape=(args.linear_num_value_heads,))
    # disable training mode -> deterministic delta-rule path
    layer.train(False)
    return layer


# --------------------------------------------------------------------------- shape

def test_shape_no_cache():
    args = toy_args()
    layer = make_layer(args)
    x = mx.random.normal((2, 5, args.hidden_size)).astype(mx.float32)
    out = layer(x)
    assert out.shape == (2, 5, args.hidden_size)
    assert not np.isnan(np.array(out)).any()


def test_shape_with_cache():
    args = toy_args()
    layer = make_layer(args)
    x = mx.random.normal((1, 6, args.hidden_size)).astype(mx.float32)
    cache = ArraysCache(size=2)
    out = layer(x, cache=cache)
    assert out.shape == (1, 6, args.hidden_size)
    # cache populated: [0] conv state [B, K-1, conv_dim], [1] recurrent state
    assert cache[0] is not None and cache[1] is not None
    assert cache[0].shape == (1, args.linear_conv_kernel_dim - 1, args.conv_dim)
    assert cache[1].shape == (
        1,
        args.linear_num_value_heads,
        args.linear_value_head_dim,
        args.linear_key_head_dim,
    )
    assert not np.isnan(np.array(out)).any()


# --------------------------------------------------------------------------- chunked == single-shot

def test_chunked_equals_single_shot():
    """24 tokens in one shot vs fed in chunks of (1,7,16), threading the cache.
    This is the conv/state-bug detector."""
    args = toy_args()
    layer = make_layer(args, seed=3)
    T = 24
    x = mx.random.normal((1, T, args.hidden_size)).astype(mx.float32)

    full = layer(x)

    cache = ArraysCache(size=2)
    outs = []
    start = 0
    for n in (1, 7, 16):
        chunk = x[:, start : start + n, :]
        outs.append(layer(chunk, cache=cache))
        start += n
    chunked = mx.concatenate(outs, axis=1)

    a = np.array(full)
    b = np.array(chunked)
    assert a.shape == b.shape == (1, T, args.hidden_size)
    assert np.allclose(a, b, atol=1e-4, rtol=1e-4), (
        f"max abs diff {np.abs(a - b).max():.3e}"
    )


# --------------------------------------------------------------------------- sigmoid gate pinning

def test_sigmoid_gate_not_silu():
    """Pin the SIGMOID output gate so nobody 'fixes' it back to silu.

    - gate = 0  ->  sigmoid(0) = 0.5  ->  output == 0.5 * rms_norm(x).
      silu(0) = 0, which would kill the output. A nonzero 0.5x output PROVES sigmoid.
    - gate = large negative  ->  sigmoid ~ 0  ->  output ~ 0.
    """
    hv = 16
    norm = Qwen4ExpRMSNormGated(hv, eps=1e-6)
    norm.weight = mx.ones((hv,))
    x = mx.random.normal((2, 3, hv)).astype(mx.float32)

    # gate = 0  ->  0.5 * rms_norm(x)
    gate0 = mx.zeros_like(x)
    out0 = norm(x, gate0)
    ref = mx.fast.rms_norm(x, mx.ones((hv,)), 1e-6) * 0.5
    assert np.allclose(np.array(out0), np.array(ref), atol=1e-5), "gate=0 must give 0.5x (sigmoid)"
    # and it is NOT ~0 (which silu(0)=0 would produce)
    assert float(mx.abs(out0).max()) > 1e-3

    # gate = large negative  ->  ~0
    gate_neg = mx.full(x.shape, -30.0)
    out_neg = norm(x, gate_neg)
    assert float(mx.abs(out_neg).max()) < 1e-4, "gate=-inf must kill output toward 0"


def test_l2_normalize_matches_reference():
    x = mx.random.normal((2, 3, 4, 8)).astype(mx.float32)
    got = np.array(_l2_normalize(x, 1e-6))
    xn = np.array(x).astype(np.float64)
    denom = 1.0 / np.sqrt((xn * xn).sum(-1, keepdims=True) + 1e-6)
    ref = xn * denom
    assert np.allclose(got, ref, atol=1e-5)


# --------------------------------------------------------------------------- no NaN under a longer run

def test_no_nan_longer_sequence():
    args = toy_args()
    layer = make_layer(args, seed=9)
    x = mx.random.normal((3, 40, args.hidden_size)).astype(mx.float32) * 3.0
    out = layer(x)
    arr = np.array(out)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


# --------------------------------------------------------------------------- readout scale

def test_delta_readout_has_inv_sqrt_head_v_dim_scale():
    """Regression: qwen4_exp GATED_DELTA_NET scales its readout by 1/sqrt(head_v_dim)
    (ggml ops.cpp:10824 `attn_data[j] = sum * scale`). mlx_lm's gated_delta_update omits
    it, so GatedDeltaNet must re-apply it before the gated norm. Missing this made the
    layer-0 output ~11x too large and every real-weight generation incoherent.

    We reconstruct the module's own pipeline twice -- once feeding the RAW (unscaled)
    delta output to the gated norm + out_proj, once feeding the 1/sqrt(head_v_dim)-scaled
    output -- and assert the module matches the SCALED variant, not the raw one.
    """
    from mlx_lm.models.gated_delta import gated_delta_update

    args = toy_args()
    layer = make_layer(args, seed=3)
    dn = layer
    x = mx.random.normal((2, 6, args.hidden_size)).astype(mx.float32)
    B, S, _ = x.shape

    qkv = dn.in_proj_qkv(x)
    z = dn.in_proj_z(x)
    b = dn.in_proj_b(x)
    a = dn.in_proj_a(x)
    conv_state = mx.zeros((B, dn.conv_kernel_size - 1, dn.conv_dim), dtype=x.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(dn.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [dn.key_dim, 2 * dn.key_dim], axis=-1),
            [dn.num_k_heads, dn.num_k_heads, dn.num_v_heads],
            [dn.head_k_dim, dn.head_k_dim, dn.head_v_dim],
        )
    ]
    q = _l2_normalize(q, dn.eps)
    k = _l2_normalize(k, dn.eps)
    raw, _ = gated_delta_update(
        q, k, v, a, b, dn.A_log, dn.dt_bias, None, None, use_kernel=not dn.training
    )

    zr = z.reshape(B, S, dn.num_v_heads, dn.head_v_dim)
    scale = 1.0 / (dn.head_v_dim ** 0.5)

    scaled = dn.out_proj(dn.norm(raw * scale, zr).reshape(B, S, dn.value_dim))
    unscaled = dn.out_proj(dn.norm(raw, zr).reshape(B, S, dn.value_dim))
    module = dn(x)

    got = np.array(module)
    assert np.allclose(got, np.array(scaled), atol=1e-4), (
        "GatedDeltaNet output must include the 1/sqrt(head_v_dim) readout scale"
    )
    # and it must NOT match the unscaled variant (the RMS norm does not fully absorb it,
    # so the two are numerically distinguishable -- this is what the bug looked like)
    assert not np.allclose(got, np.array(unscaled), atol=1e-4), (
        "scaled and unscaled outputs are indistinguishable -- test cannot catch the bug"
    )
    # the exposed constant must be exactly 1/sqrt(head_v_dim)
    assert abs(dn.delta_out_scale - 1.0 / (dn.head_v_dim ** 0.5)) < 1e-12
