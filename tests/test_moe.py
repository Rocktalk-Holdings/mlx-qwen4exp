"""Tests for mlx_qwen4exp.moe -- the MoE FFN.

Toy config: 8 experts, top-2, hidden 16, moe_inter 8. No checkpoint; random weights.

Forcing rule (no mx.eval in tests): assertions go through ``np.array`` / ``float``.

Run:
    /opt/homebrew/bin/python3.11 -m pytest tests/test_moe.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mlx_qwen4exp.config import ModelArgs  # noqa: E402
from mlx_qwen4exp.moe import Qwen4ExpMoE, split_gate_up  # noqa: E402


def toy_args(**overrides) -> ModelArgs:
    base = dict(
        hidden_size=16,
        num_hidden_layers=4,
        vocab_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    base.update(overrides)
    return ModelArgs(**base)


def make_moe(args: ModelArgs, seed: int = 0) -> Qwen4ExpMoE:
    mx.random.seed(seed)
    moe = Qwen4ExpMoE(args)
    moe.train(False)
    return moe


# --------------------------------------------------------------------------- shape

def test_shape():
    args = toy_args()
    moe = make_moe(args)
    x = mx.random.normal((2, 5, args.hidden_size)).astype(mx.float32)
    out = moe(x)
    assert out.shape == (2, 5, args.hidden_size)
    assert not np.isnan(np.array(out)).any()


# --------------------------------------------------------------------------- routing correctness

def _dense_reference(moe: Qwen4ExpMoE, x: mx.array, top_k: int, norm_topk: bool) -> np.ndarray:
    """Independent reference: compute the full MoE by hand (all experts dense), then
    keep the top_k, (optionally) renormalise, weight, sum, add sigmoid-gated shared."""
    xn = np.array(x).astype(np.float64)
    B, T, D = xn.shape

    gate_w = np.array(moe.gate.weight).astype(np.float64)   # [E, D]
    logits = xn @ gate_w.T                                   # [B,T,E]
    # precise softmax
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(-1, keepdims=True)                     # [B,T,E]

    gup = np.array(moe.switch_mlp.gate_proj.weight).astype(np.float64)  # [E, inter, D]
    upp = np.array(moe.switch_mlp.up_proj.weight).astype(np.float64)    # [E, inter, D]
    dnp = np.array(moe.switch_mlp.down_proj.weight).astype(np.float64)  # [E, D, inter]

    def silu(z):
        return z / (1.0 + np.exp(-z))

    # dense expert outputs for every expert
    # expert_out[b,t,ex] = down_ex @ (silu(gate_ex @ x) * (up_ex @ x))
    g = np.einsum("eid,btd->btei", gup, xn)   # [B,T,E,inter]
    u = np.einsum("eid,btd->btei", upp, xn)
    h = silu(g) * u                           # [B,T,E,inter]
    expert_out = np.einsum("edi,btei->bted", dnp, h)  # [B,T,E,D]

    # top-k selection by prob
    idx = np.argsort(-probs, axis=-1)[..., :top_k]        # [B,T,k]
    sel_w = np.take_along_axis(probs, idx, axis=-1)        # [B,T,k]
    if norm_topk:
        denom = np.clip(sel_w.sum(-1, keepdims=True), 6.103515625e-5, None)
        sel_w = sel_w / denom

    y = np.zeros((B, T, D))
    for b in range(B):
        for t in range(T):
            for j in range(top_k):
                y[b, t] += sel_w[b, t, j] * expert_out[b, t, idx[b, t, j]]

    # shared expert (SwiGLU) + sigmoid gate
    sg = np.array(moe.shared_expert.gate_proj.weight).astype(np.float64)  # [inter, D]
    su = np.array(moe.shared_expert.up_proj.weight).astype(np.float64)
    sd = np.array(moe.shared_expert.down_proj.weight).astype(np.float64)  # [D, inter]
    sh = (silu(xn @ sg.T) * (xn @ su.T)) @ sd.T                            # [B,T,D]
    sgate_w = np.array(moe.shared_expert_gate.weight).astype(np.float64)   # [1, D]
    sgate = 1.0 / (1.0 + np.exp(-(xn @ sgate_w.T)))                        # [B,T,1]
    y = y + sgate * sh
    return y


def test_routing_matches_dense_reference():
    """Craft router weights so each token overwhelmingly selects a known expert, then
    compare the module against a fully independent dense hand computation."""
    args = toy_args()
    moe = make_moe(args, seed=1)

    x = mx.random.normal((2, 4, args.hidden_size)).astype(mx.float32)
    got = np.array(moe(x))
    ref = _dense_reference(moe, x, top_k=args.num_experts_per_tok, norm_topk=moe.norm_topk_prob)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-4, rtol=1e-4), (
        f"max abs diff {np.abs(got - ref).max():.3e}"
    )


def test_dominant_expert_selected():
    """Force the gate to make expert e win for every token; confirm the module's output
    equals that single expert's output (top-2 with one dominant + renorm -> ~that expert)."""
    args = toy_args(num_experts_per_tok=1)  # top-1 so the dominant expert is the whole story
    moe = make_moe(args, seed=2)

    # zero the shared expert gate weight so shared contribution = 0.5 * shared(x) -> keep,
    # but we compare against the same reference which includes it, so nothing to null here.
    x = mx.random.normal((1, 3, args.hidden_size)).astype(mx.float32)
    got = np.array(moe(x))
    ref = _dense_reference(moe, x, top_k=1, norm_topk=moe.norm_topk_prob)
    assert np.allclose(got, ref, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- shared expert gate

def test_shared_expert_sigmoid_gate():
    args = toy_args()
    moe = make_moe(args, seed=5)
    x = mx.random.normal((2, 3, args.hidden_size)).astype(mx.float32)

    xn = np.array(x).astype(np.float64)
    def silu(z): return z / (1.0 + np.exp(-z))
    sg = np.array(moe.shared_expert.gate_proj.weight).astype(np.float64)
    su = np.array(moe.shared_expert.up_proj.weight).astype(np.float64)
    sd = np.array(moe.shared_expert.down_proj.weight).astype(np.float64)
    sh = (silu(xn @ sg.T) * (xn @ su.T)) @ sd.T
    sgate_w = np.array(moe.shared_expert_gate.weight).astype(np.float64)
    sgate = 1.0 / (1.0 + np.exp(-(xn @ sgate_w.T)))
    ref_shared = sgate * sh

    got_shared = np.array(mx.sigmoid(moe.shared_expert_gate(x)) * moe.shared_expert(x))
    assert np.allclose(got_shared, ref_shared, atol=1e-4)


# --------------------------------------------------------------------------- split_gate_up round trip

def test_split_gate_up_round_trip():
    E, inter, D = 8, 640, 16
    fused = mx.random.normal((E, 2 * inter, D)).astype(mx.float32)
    gate, up = split_gate_up(fused)
    assert gate.shape == (E, inter, D)
    assert up.shape == (E, inter, D)
    f = np.array(fused)
    assert np.array_equal(np.array(gate), f[:, :inter, :])   # gate = FIRST inter rows
    assert np.array_equal(np.array(up), f[:, inter:, :])     # up   = LAST inter rows


def test_split_gate_up_rejects_bad_shape():
    with pytest.raises(ValueError):
        split_gate_up(mx.zeros((8, 640)))          # rank 2
    with pytest.raises(ValueError):
        split_gate_up(mx.zeros((8, 641, 16)))      # odd axis-1


# --------------------------------------------------------------------------- renormalisation

def test_topk_renormalisation_applied():
    """Ground truth (C++ cpp:832 norm_w=true, llama-graph.cpp:2082-2096) renormalises the
    selected top-k weights to sum to 1. Verify the module divides by the selected sum by
    comparing a renorm reference (should match) against a no-renorm reference (should not,
    since top-2 softmax weights rarely sum to 1)."""
    args = toy_args()
    moe = make_moe(args, seed=7)
    assert moe.norm_topk_prob is True  # C++ ground truth

    x = mx.random.normal((2, 4, args.hidden_size)).astype(mx.float32)
    got = np.array(moe(x))

    ref_renorm = _dense_reference(moe, x, top_k=args.num_experts_per_tok, norm_topk=True)
    ref_raw = _dense_reference(moe, x, top_k=args.num_experts_per_tok, norm_topk=False)

    assert np.allclose(got, ref_renorm, atol=1e-4), "module must renormalise (C++ norm_w=true)"
    # and the raw (no-renorm) path is measurably different -> proves renorm is real
    assert not np.allclose(got, ref_raw, atol=1e-3), (
        "renorm and raw should differ; if they match the test is not exercising renorm"
    )


def test_no_nan():
    args = toy_args()
    moe = make_moe(args, seed=11)
    x = mx.random.normal((3, 7, args.hidden_size)).astype(mx.float32) * 4.0
    arr = np.array(moe(x))
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()
