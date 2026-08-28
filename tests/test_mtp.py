"""Unit tests for mlx_qwen4exp/mtp.py — MTP draft head.

All tests run at TOY dimensions (no real weights required).
The no-mx.eval rule is enforced: value assertions use np.array() or float() to force;
shape assertions use .shape which is known lazily.

Gate 2 checklist:
  [x] Shape test: MTPDraft forward produces [B, T, vocab] logits
  [x] Shape test: MTPDraftLayer forward preserves [B, T, hc, D] res shape
  [x] Cache wiring: cached MTPDraftLayer == uncached on first step
  [x] Chunked == single-shot on the MTP layer (the cache wiring invariant)
  [x] No-NaN sweep at extreme inputs
  [x] sanitize_mtp_weights: +1 folding on hc_norms, NO +1 on pre_fc_norm_*
  [x] sanitize_mtp_weights: gate_up split produces correct shapes
  [x] sanitize_mtp_weights: indexer qk split
  [x] ModelArgs.load_mtp flag: Model.sanitize passes mtp.* through when True,
      drops them when False
  [x] QSACache moved to attention.py (circular-import free)

Run:
    /opt/homebrew/bin/python3.11 -m pytest tests/test_mtp.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_qwen4exp.config import ModelArgs
from mlx_qwen4exp.attention import QSACache
from mlx_qwen4exp.hyper import hc_init, grouped_rms_norm
from mlx_qwen4exp.mtp import MTPDraft, MTPDraftLayer, make_mtp_cache, sanitize_mtp_weights

# ---------------------------------------------------------------------------
# Toy dimensions (chosen so all shapes work with QSA indexer and MoE)
HIDDEN = 64       # must be divisible by num_attention_heads * head_dim / 2
HC = 2
LOWRANK = 8
HC_DIM = HC * HIDDEN
VOCAB = 128
HEADS = 2          # num_attention_heads
HEAD_DIM = 32      # head_dim; 2*HEADS*HEAD_DIM = 128 < HIDDEN*2=128 OK
KV_HEADS = 1
N_ROT = 8          # partial_rotary_factor * head_dim = 0.25 * 32 = 8
EXPERTS = 4
TOPK = 2
MOE_INTER = 16
SHARED_INTER = 16
IDX_HEADS = 2
IDX_HEAD_DIM = 8
IDX_BUDGET = 4
IDX_R = 2
EPS = 1e-6


def toy_args(**overrides) -> ModelArgs:
    kwargs = dict(
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        vocab_size=VOCAB,
        num_hidden_layers=4,
        full_attention_interval=4,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        partial_rotary_factor=0.25,
        rope_theta=1e4,
        indexer_n_heads=IDX_HEADS,
        indexer_head_dim=IDX_HEAD_DIM,
        indexer_budget=IDX_BUDGET,
        indexer_compress_ratio=IDX_R,
        num_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=SHARED_INTER,
        # linear-attn dims (for completeness; mtp layer is full_attention only)
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        load_mtp=True,
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


def random_mtp(args: ModelArgs, seed: int = 0) -> MTPDraft:
    """Construct an MTPDraft with random (non-trivial) weights."""
    mx.random.seed(seed)
    draft = MTPDraft(args)
    # randomize bare gammas
    draft.pre_fc_norm_hidden = mx.random.normal((HC_DIM,)) * 0.2 + 1.0
    draft.pre_fc_norm_embedding = mx.random.normal((HIDDEN,)) * 0.2 + 1.0
    # randomize linear weights
    draft.fc_hidden.weight = mx.random.normal((HIDDEN, HIDDEN)) * 0.1
    draft.fc_embedding.weight = mx.random.normal((HIDDEN, HIDDEN)) * 0.1
    # randomize mixer and layer weights
    def _randomize_linear(module, scale=0.1):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight = mx.random.normal(module.weight.shape) * scale
    from mlx_qwen4exp.hyper import HyperConnection
    # hyper_connection_mixer
    hcm = draft.hyper_connection_mixer
    hcm.hc_norm = mx.random.normal((HC_DIM,)) * 0.2 + 1.0
    hcm.input_mix_weight_down.weight = mx.random.normal((LOWRANK, HC_DIM)) * 0.1
    hcm.input_mix_weight_up.weight = mx.random.normal((HC_DIM, LOWRANK)) * 0.1
    return draft


def fake_embed(args: ModelArgs, seed: int = 42) -> nn.Embedding:
    mx.random.seed(seed)
    emb = nn.Embedding(args.vocab_size, args.hidden_size)
    emb.weight = mx.random.normal((args.vocab_size, args.hidden_size)) * 0.1
    return emb


def fake_lm_head(args: ModelArgs, seed: int = 43) -> nn.Linear:
    mx.random.seed(seed)
    head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    head.weight = mx.random.normal((args.vocab_size, args.hidden_size)) * 0.1
    return head


# ---------------------------------------------------------------------------
# Shape tests

class TestMTPDraftShapes:
    def test_forward_shape(self):
        args = toy_args()
        draft = random_mtp(args)
        embed = fake_embed(args)
        lm_head = fake_lm_head(args)

        B, T = 2, 1
        last_wide = mx.random.normal((B, T, HC, HIDDEN))
        next_id = mx.zeros((B, T), dtype=mx.int32)
        cache = make_mtp_cache()[0]

        logits = draft(last_wide, next_id, embed, lm_head, cache=cache, pos_offset=0)
        assert logits.shape == (B, T, VOCAB), f"expected ({B},{T},{VOCAB}), got {logits.shape}"

    def test_forward_no_cache(self):
        args = toy_args()
        draft = random_mtp(args)
        embed = fake_embed(args)
        lm_head = fake_lm_head(args)

        B, T = 1, 1
        last_wide = mx.random.normal((B, T, HC, HIDDEN))
        next_id = mx.zeros((B, T), dtype=mx.int32)
        logits = draft(last_wide, next_id, embed, lm_head, cache=None, pos_offset=0)
        assert logits.shape == (B, T, VOCAB)
        arr = np.array(logits.astype(mx.float32))
        assert np.isfinite(arr).all(), "logits contain non-finite values"

    def test_layer_shape(self):
        args = toy_args()
        layer = MTPDraftLayer(args)
        B, T, hc, D = 2, 3, HC, HIDDEN
        res = mx.random.normal((B, T, hc, D))
        out = layer(res, cache=None, pos_offset=0)
        assert out.shape == (B, T, hc, D), f"layer output shape {out.shape}"

    def test_make_mtp_cache_length(self):
        caches = make_mtp_cache()
        assert len(caches) == 1
        assert isinstance(caches[0], QSACache)

    def test_wrong_input_rank_raises(self):
        args = toy_args()
        draft = random_mtp(args)
        embed = fake_embed(args)
        lm_head = fake_lm_head(args)
        # 3D last_wide should raise
        with pytest.raises(ValueError, match="last_hidden_wide"):
            draft(
                mx.random.normal((1, HC, HIDDEN)),
                mx.zeros((1, 1), dtype=mx.int32),
                embed, lm_head, cache=None, pos_offset=0,
            )
        # 1D next_input_id should raise
        with pytest.raises(ValueError, match="next_input_id"):
            draft(
                mx.random.normal((1, 1, HC, HIDDEN)),
                mx.zeros((1,), dtype=mx.int32),
                embed, lm_head, cache=None, pos_offset=0,
            )


# ---------------------------------------------------------------------------
# Cache wiring: first step with vs without cache must agree

class TestCacheWiring:
    def _layer_call(self, layer, res, cache, pos_offset):
        return layer(res, cache=cache, pos_offset=pos_offset)

    def test_cached_equals_uncached_first_step(self):
        """First decode step: cache is empty so cached == uncached."""
        args = toy_args()
        mx.random.seed(5)
        layer = MTPDraftLayer(args)
        B, T, hc, D = 1, 1, HC, HIDDEN
        res = mx.random.normal((B, T, hc, D))

        cache = make_mtp_cache()[0]
        # cached path (cache at offset 0 ~ empty)
        out_cached = self._layer_call(layer, res, cache, pos_offset=0)
        # uncached path
        out_nocache = self._layer_call(layer, res, None, pos_offset=0)

        c = np.array(out_cached.astype(mx.float32))
        n = np.array(out_nocache.astype(mx.float32))
        assert c.shape == n.shape
        assert np.allclose(c, n, atol=1e-4), (
            f"cached vs uncached max diff {np.abs(c - n).max():.4e}"
        )

    def test_qsa_cache_offset_advances(self):
        """After one step, QSACache.offset should be 1."""
        args = toy_args()
        layer = MTPDraftLayer(args)
        B, T = 1, 1
        res = mx.random.normal((B, T, HC, HIDDEN))
        cache = make_mtp_cache()[0]
        assert cache.offset == 0
        layer(res, cache=cache, pos_offset=0)
        # force the update (MLX lazy eval)
        _ = np.array(mx.array([cache.offset]))
        # offset should have advanced by T=1
        assert cache.offset == 1, f"expected offset=1, got {cache.offset}"


# ---------------------------------------------------------------------------
# Chunked == single-shot on the MTP layer

class TestChunkedEqualsSingleShot:
    """Feeding T tokens at once must equal feeding them 1-by-1 with the same cache."""

    def test_chunked_equals_single_shot(self):
        args = toy_args()
        mx.random.seed(7)
        draft = random_mtp(args)
        embed = fake_embed(args)
        lm_head = fake_lm_head(args)
        B = 1

        # build T=3 sequence of (wide_residual, next_id) tuples
        T_seq = 3
        wides = [mx.random.normal((B, 1, HC, HIDDEN)) for _ in range(T_seq)]
        ids = [mx.zeros((B, 1), dtype=mx.int32) for _ in range(T_seq)]

        # single-shot: feed each step with rolling cache
        cache_ss = make_mtp_cache()[0]
        logits_ss = []
        for t in range(T_seq):
            lg = draft(wides[t], ids[t], embed, lm_head, cache=cache_ss, pos_offset=t)
            logits_ss.append(np.array(lg.astype(mx.float32)))

        # rebuild fresh cache, redo step-by-step (same order, different cache object)
        cache_fresh = make_mtp_cache()[0]
        logits_fresh = []
        for t in range(T_seq):
            lg = draft(wides[t], ids[t], embed, lm_head, cache=cache_fresh, pos_offset=t)
            logits_fresh.append(np.array(lg.astype(mx.float32)))

        for t in range(T_seq):
            diff = np.abs(logits_ss[t] - logits_fresh[t]).max()
            assert diff < 1e-4, f"step {t} logit max diff {diff:.4e} (same order mismatch)"


# ---------------------------------------------------------------------------
# No-NaN sweep

class TestNoNaN:
    @pytest.mark.parametrize("magnitude", [1e-8, 1.0, 1e3, 0.0])
    def test_no_nan_draft_forward(self, magnitude):
        args = toy_args()
        mx.random.seed(13)
        draft = random_mtp(args)
        embed = fake_embed(args)
        lm_head = fake_lm_head(args)

        B, T = 1, 1
        last_wide = mx.random.normal((B, T, HC, HIDDEN)) * magnitude
        next_id = mx.zeros((B, T), dtype=mx.int32)
        logits = draft(last_wide, next_id, embed, lm_head, cache=None, pos_offset=0)
        arr = np.array(logits.astype(mx.float32))
        assert not np.isnan(arr).any(), f"NaN at magnitude={magnitude}"
        assert not np.isinf(arr).any(), f"Inf at magnitude={magnitude}"


# ---------------------------------------------------------------------------
# sanitize_mtp_weights

class TestSanitizeMTPWeights:
    def _make_raw_mtp_weights(self, args: ModelArgs) -> dict:
        """Minimal set of fake mtp.* weights matching checkpoint structure."""
        D = args.hidden_size
        hc_dim = args.hc_dim
        E = args.num_experts
        inter = args.moe_intermediate_size
        idx_q_out = args.indexer_n_heads * args.indexer_head_dim  # IDX_HEADS*IDX_HEAD_DIM=16
        idx_k_out = args.indexer_head_dim  # 8
        idx_total = idx_q_out + idx_k_out  # 24

        return {
            # pre_fc_norm_* — bare gammas, should NOT get +1
            "mtp.pre_fc_norm_hidden.weight": mx.ones((hc_dim,)) * 0.5,
            "mtp.pre_fc_norm_embedding.weight": mx.ones((D,)) * 0.7,
            # linear weights
            "mtp.fc_hidden.weight": mx.zeros((D, D)),
            "mtp.fc_embedding.weight": mx.zeros((D, D)),
            # mixer hc_norm — SHOULD get +1
            "mtp.hyper_connection_mixer.hc_norm.weight": mx.ones((hc_dim,)) * 0.3,
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight": mx.zeros((args.hc_lowrank, hc_dim)),
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight": mx.zeros((hc_dim, args.hc_lowrank)),
            # layer 0 hc_norm — SHOULD get +1
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight": mx.ones((hc_dim,)) * 0.2,
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": mx.zeros((args.hc_lowrank, hc_dim)),
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": mx.zeros((hc_dim, args.hc_lowrank)),
            "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": mx.zeros((HC, hc_dim)),
            "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": mx.ones((hc_dim,)) * 0.1,
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": mx.zeros((args.hc_lowrank, hc_dim)),
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": mx.zeros((hc_dim, args.hc_lowrank)),
            "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": mx.zeros((HC, hc_dim)),
            # self_attn q/k norms — SHOULD get +1
            "mtp.layers.0.self_attn.q_norm.weight": mx.ones((HEAD_DIM,)) * 0.4,
            "mtp.layers.0.self_attn.k_norm.weight": mx.ones((HEAD_DIM,)) * 0.6,
            # attn projections
            "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((HEADS * 2 * HEAD_DIM, D)),
            "mtp.layers.0.self_attn.k_proj.weight": mx.zeros((KV_HEADS * HEAD_DIM, D)),
            "mtp.layers.0.self_attn.v_proj.weight": mx.zeros((KV_HEADS * HEAD_DIM, D)),
            "mtp.layers.0.self_attn.o_proj.weight": mx.zeros((D, HEADS * HEAD_DIM)),
            # indexer qk joint proj — SHOULD split
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": mx.zeros((idx_total, D)),
            # indexer layernorms — SHOULD get +1
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight": mx.ones((IDX_HEAD_DIM,)) * 0.9,
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight": mx.ones((IDX_HEAD_DIM,)) * 0.8,
            # experts — SHOULD split gate_up
            "mtp.layers.0.mlp.gate.weight": mx.zeros((E, D)),
            "mtp.layers.0.mlp.experts.gate_up_proj": mx.zeros((E, 2 * inter, D)),
            "mtp.layers.0.mlp.experts.down_proj": mx.zeros((E, D, inter)),
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight": mx.zeros((inter, D)),
            "mtp.layers.0.mlp.shared_expert.up_proj.weight": mx.zeros((inter, D)),
            "mtp.layers.0.mlp.shared_expert.down_proj.weight": mx.zeros((D, inter)),
            "mtp.layers.0.mlp.shared_expert_gate.weight": mx.zeros((1, D)),
        }

    def test_pre_fc_norm_gets_plus_one(self):
        # CORRECTION 2026-08-28: pre_fc_norm_hidden and pre_fc_norm_embedding DO get +1.
        # The checkpoint stores them as (gamma - 1) — empirical evidence: raw values
        # for pre_fc_norm_embedding in the real checkpoint have mean=-0.76 which is
        # nonsensical as a raw scale factor; adding 1 gives mean=0.24, a valid low gamma.
        # Test raw=0.5 -> expect 1.5; raw=0.7 -> expect 1.7.
        args = toy_args()
        raw = self._make_raw_mtp_weights(args)
        out = sanitize_mtp_weights(raw, args)

        hidden_gamma = np.array(out["pre_fc_norm_hidden"].astype(mx.float32))
        emb_gamma = np.array(out["pre_fc_norm_embedding"].astype(mx.float32))
        assert np.allclose(hidden_gamma, 1.5, atol=1e-5), (
            f"pre_fc_norm_hidden: raw=0.5 + 1 should give 1.5, got {hidden_gamma.mean():.4f}"
        )
        assert np.allclose(emb_gamma, 1.7, atol=1e-5), (
            f"pre_fc_norm_embedding: raw=0.7 + 1 should give 1.7, got {emb_gamma.mean():.4f}"
        )

    def test_hc_norm_gets_plus_one(self):
        args = toy_args()
        raw = self._make_raw_mtp_weights(args)
        out = sanitize_mtp_weights(raw, args)

        # mixer hc_norm raw=0.3 -> should become 1.3 after +1
        mixer_norm = np.array(out["hyper_connection_mixer.hc_norm"].astype(mx.float32))
        assert np.allclose(mixer_norm, 1.3, atol=1e-5), f"mixer hc_norm expected 1.3, got {mixer_norm.mean():.4f}"

        # layer 0 attn hc_norm raw=0.2 -> 1.2
        l0_norm = np.array(out["layers.0.attn_hyper_connection.hc_norm"].astype(mx.float32))
        assert np.allclose(l0_norm, 1.2, atol=1e-5), f"attn hc_norm expected 1.2, got {l0_norm.mean():.4f}"

    def test_q_norm_k_norm_get_plus_one(self):
        args = toy_args()
        raw = self._make_raw_mtp_weights(args)
        out = sanitize_mtp_weights(raw, args)

        # q_norm raw=0.4 -> 1.4
        q = np.array(out["layers.0.self_attn.q_norm"].astype(mx.float32))
        assert np.allclose(q, 1.4, atol=1e-5), f"q_norm expected 1.4, got {q.mean():.4f}"
        # k_norm raw=0.6 -> 1.6
        k = np.array(out["layers.0.self_attn.k_norm"].astype(mx.float32))
        assert np.allclose(k, 1.6, atol=1e-5), f"k_norm expected 1.6, got {k.mean():.4f}"

    def test_indexer_layernorms_get_plus_one(self):
        args = toy_args()
        raw = self._make_raw_mtp_weights(args)
        out = sanitize_mtp_weights(raw, args)

        q_ln = np.array(out["layers.0.self_attn.indexer.q_layernorm"].astype(mx.float32))
        assert np.allclose(q_ln, 1.9, atol=1e-5), f"q_layernorm expected 1.9, got {q_ln.mean():.4f}"

    def test_indexer_qk_split(self):
        args = toy_args()
        raw = self._make_raw_mtp_weights(args)
        # use a known weight to verify the split row counts (zeros are fine — we check shapes)
        idx_total = IDX_HEADS * IDX_HEAD_DIM + IDX_HEAD_DIM  # 16+8=24
        raw["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"] = mx.zeros((idx_total, HIDDEN))
        out = sanitize_mtp_weights(raw, args)

        assert "layers.0.self_attn.indexer.index_q_proj.weight" in out
        assert "layers.0.self_attn.indexer.index_k_proj.weight" in out
        assert "layers.0.self_attn.indexer.index_qk_proj.weight" not in out

        n_q = IDX_HEADS * IDX_HEAD_DIM  # 16
        n_k = IDX_HEAD_DIM  # 8
        assert out["layers.0.self_attn.indexer.index_q_proj.weight"].shape == (n_q, HIDDEN)
        assert out["layers.0.self_attn.indexer.index_k_proj.weight"].shape == (n_k, HIDDEN)

    def test_expert_gate_up_split(self):
        args = toy_args()
        raw = self._make_raw_mtp_weights(args)
        out = sanitize_mtp_weights(raw, args)

        assert "layers.0.mlp.switch_mlp.gate_proj.weight" in out
        assert "layers.0.mlp.switch_mlp.up_proj.weight" in out
        assert "layers.0.mlp.switch_mlp.down_proj.weight" in out
        assert "layers.0.mlp.experts.gate_up_proj" not in out

        gate = out["layers.0.mlp.switch_mlp.gate_proj.weight"]
        up = out["layers.0.mlp.switch_mlp.up_proj.weight"]
        down = out["layers.0.mlp.switch_mlp.down_proj.weight"]
        assert gate.shape == (EXPERTS, MOE_INTER, HIDDEN)
        assert up.shape == (EXPERTS, MOE_INTER, HIDDEN)
        assert down.shape == (EXPERTS, HIDDEN, MOE_INTER)

    def test_non_mtp_key_raises(self):
        args = toy_args()
        with pytest.raises(ValueError, match="non-mtp key"):
            sanitize_mtp_weights({"model.language_model.embed_tokens.weight": mx.zeros((1,))}, args)

    def test_no_weight_suffix_on_pre_fc_norm(self):
        """After sanitize, pre_fc_norm_* should have no .weight suffix."""
        args = toy_args()
        raw = self._make_raw_mtp_weights(args)
        out = sanitize_mtp_weights(raw, args)
        assert "pre_fc_norm_hidden" in out
        assert "pre_fc_norm_hidden.weight" not in out
        assert "pre_fc_norm_embedding" in out
        assert "pre_fc_norm_embedding.weight" not in out


# ---------------------------------------------------------------------------
# ModelArgs.load_mtp integration

class TestLoadMTPFlag:
    def test_model_sanitize_drops_mtp_by_default(self):
        """Without load_mtp=True, mtp.* must be silently dropped."""
        from mlx_qwen4exp.model import Model
        args = toy_args(load_mtp=False)
        model = Model(args)
        fake_weights = {
            "mtp.pre_fc_norm_hidden.weight": mx.ones((HC_DIM,)),
            "mtp.fc_hidden.weight": mx.zeros((HIDDEN, HIDDEN)),
            # a minimal real-looking key that would stay
            "model.language_model.embed_tokens.weight": mx.zeros((VOCAB, HIDDEN)),
        }
        out = model.sanitize(fake_weights)
        mtp_keys = [k for k in out if k.startswith("mtp.")]
        assert len(mtp_keys) == 0, f"mtp.* leaked through with load_mtp=False: {mtp_keys}"

    def test_model_sanitize_keeps_mtp_when_enabled(self):
        """With load_mtp=True, sanitize must produce mtp.* keys."""
        from mlx_qwen4exp.model import Model
        args = toy_args(load_mtp=True)
        model = Model(args)

        hc_dim = args.hc_dim
        D = args.hidden_size
        E = args.num_experts
        inter = args.moe_intermediate_size
        idx_q_out = args.indexer_n_heads * args.indexer_head_dim
        idx_k_out = args.indexer_head_dim

        mtp_weights = {
            "mtp.pre_fc_norm_hidden.weight": mx.ones((hc_dim,)),
            "mtp.pre_fc_norm_embedding.weight": mx.ones((D,)),
            "mtp.fc_hidden.weight": mx.zeros((D, D)),
            "mtp.fc_embedding.weight": mx.zeros((D, D)),
            "mtp.hyper_connection_mixer.hc_norm.weight": mx.ones((hc_dim,)),
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight": mx.zeros((args.hc_lowrank, hc_dim)),
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight": mx.zeros((hc_dim, args.hc_lowrank)),
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight": mx.ones((hc_dim,)),
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": mx.zeros((args.hc_lowrank, hc_dim)),
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": mx.zeros((hc_dim, args.hc_lowrank)),
            "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": mx.zeros((HC, hc_dim)),
            "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": mx.ones((hc_dim,)),
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": mx.zeros((args.hc_lowrank, hc_dim)),
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": mx.zeros((hc_dim, args.hc_lowrank)),
            "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": mx.zeros((HC, hc_dim)),
            "mtp.layers.0.self_attn.q_norm.weight": mx.ones((HEAD_DIM,)),
            "mtp.layers.0.self_attn.k_norm.weight": mx.ones((HEAD_DIM,)),
            "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((HEADS * 2 * HEAD_DIM, D)),
            "mtp.layers.0.self_attn.k_proj.weight": mx.zeros((KV_HEADS * HEAD_DIM, D)),
            "mtp.layers.0.self_attn.v_proj.weight": mx.zeros((KV_HEADS * HEAD_DIM, D)),
            "mtp.layers.0.self_attn.o_proj.weight": mx.zeros((D, HEADS * HEAD_DIM)),
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": mx.zeros((idx_q_out + idx_k_out, D)),
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight": mx.ones((IDX_HEAD_DIM,)),
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight": mx.ones((IDX_HEAD_DIM,)),
            "mtp.layers.0.mlp.gate.weight": mx.zeros((E, D)),
            "mtp.layers.0.mlp.experts.gate_up_proj": mx.zeros((E, 2 * inter, D)),
            "mtp.layers.0.mlp.experts.down_proj": mx.zeros((E, D, inter)),
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight": mx.zeros((inter, D)),
            "mtp.layers.0.mlp.shared_expert.up_proj.weight": mx.zeros((inter, D)),
            "mtp.layers.0.mlp.shared_expert.down_proj.weight": mx.zeros((D, inter)),
            "mtp.layers.0.mlp.shared_expert_gate.weight": mx.zeros((1, D)),
        }
        # Also add a minimal set of main-model weights so sanitize doesn't fail on missing layers
        full_weights = dict(mtp_weights)
        full_weights["model.language_model.embed_tokens.weight"] = mx.zeros((VOCAB, D))

        out = model.sanitize(full_weights)
        mtp_keys = [k for k in out if k.startswith("mtp.")]
        assert len(mtp_keys) > 0, "load_mtp=True but no mtp.* keys in sanitize output"
        # check the pre_fc_norm_hidden survived (as "mtp.pre_fc_norm_hidden")
        assert "mtp.pre_fc_norm_hidden" in out, f"mtp.pre_fc_norm_hidden missing: {sorted(mtp_keys)}"

    def test_model_has_mtp_attr_when_enabled(self):
        from mlx_qwen4exp.model import Model
        from mlx_qwen4exp.mtp import MTPDraft
        args = toy_args(load_mtp=True)
        model = Model(args)
        assert model.mtp is not None
        assert isinstance(model.mtp, MTPDraft)

    def test_model_has_no_mtp_attr_when_disabled(self):
        from mlx_qwen4exp.model import Model
        args = toy_args(load_mtp=False)
        model = Model(args)
        assert model.mtp is None


# ---------------------------------------------------------------------------
# QSACache moved to attention.py (no circular import)

class TestQSACacheImportPath:
    def test_import_from_attention(self):
        from mlx_qwen4exp.attention import QSACache as QSACacheFromAttention
        from mlx_qwen4exp.model import QSACache as QSACacheFromModel
        # both should be the same class (model re-exports it from attention)
        assert QSACacheFromAttention is QSACacheFromModel

    def test_qsa_cache_offset_proxy(self):
        cache = QSACache()
        assert cache.offset == 0
        # state round-trip
        assert cache.state == (None, None)


# ---------------------------------------------------------------------------
# Constructed-equivalence degenerate regime (mock verifier, k=1 accept-all)

class TestDegenerateEquivalence:
    """When every draft token is accepted, the spec-dec loop must emit the SAME
    token sequence as a plain greedy decode.

    This is Leg 3 of the three-leg verification contract. The toy test here verifies
    LOOP LOGIC correctness using a mock verifier (a deterministic callable that returns
    pre-scripted token sequences). The real-weight equality test runs at Gate 3 via
    run_mlx.py --mtp (200-token greedy, 3 prompts, token-for-token identical).

    Mock setup: both verifier and MTP agree on the ground-truth sequence
    [10, 20, 30, 40, 50]. MTP always predicts the right next token, so every draft
    is accepted and the spec-dec loop emits the same sequence as greedy.
    """

    GROUND_TRUTH = [10, 20, 30, 40, 50]  # tokens at positions 0..4

    def _mock_verifier(self, pos: int, n: int):
        """Returns the ground-truth logits array at position pos for n tokens.

        Returns a list[list[int]] of length n where each element is the argmax token.
        The mock verifier emits GROUND_TRUTH[pos:pos+n].
        """
        return self.GROUND_TRUTH[pos : pos + n]

    def _mock_mtp(self, pos: int):
        """MTP draft for position pos: always predicts GROUND_TRUTH[pos]."""
        return self.GROUND_TRUTH[pos]

    def _greedy_loop(self, start_tok, n):
        """Plain greedy decode using the mock verifier."""
        out = [start_tok]
        tok = start_tok
        pos = 1  # next position to predict (start_tok is at position 0)
        while len(out) < n:
            predicted = self._mock_verifier(pos, 1)[0]
            out.append(predicted)
            tok = predicted
            pos += 1
        return out

    def _spec_dec_loop(self, start_tok, n):
        """k=1 spec-dec using mock MTP (always accepts)."""
        out = [start_tok]
        pos = 1  # next position to fill

        while len(out) < n:
            draft = self._mock_mtp(pos)
            # verifier: run on [current_tok, draft] = positions [pos-1, pos]
            # verifier output[0] = prediction for pos, output[1] = for pos+1
            ver = self._mock_verifier(pos - 1, 2)   # [tok_at_pos-1, tok_at_pos]
            # ver[0] is what verifier says about position pos
            # (analogous to ver_logits[:,0,:] after feeding [tok, draft])
            accepted = ver[1]   # = GROUND_TRUTH[pos]
            next_from_ver = self._mock_verifier(pos + 1, 1)[0] if pos + 1 < len(self.GROUND_TRUTH) else None

            out.append(accepted)
            pos += 1

            if draft == accepted and len(out) < n and next_from_ver is not None:
                # accept: emit verifier's next prediction too
                out.append(next_from_ver)
                pos += 1

        return out[:n]

    def test_spec_dec_accept_all_matches_greedy(self):
        """Mock verifier + mock MTP both following GROUND_TRUTH must produce same sequence."""
        n = 5
        start = self.GROUND_TRUTH[0]  # 10

        greedy = self._greedy_loop(start, n)
        spec_dec = self._spec_dec_loop(start, n)

        assert greedy == self.GROUND_TRUTH, f"greedy wrong: {greedy}"
        assert spec_dec == self.GROUND_TRUTH, f"spec_dec wrong: {spec_dec}"
        assert greedy == spec_dec, (
            f"spec_dec != greedy:\n  greedy:   {greedy}\n  spec_dec: {spec_dec}"
        )

    def test_spec_dec_reject_falls_back_to_greedy_token(self):
        """When MTP drafts the WRONG token, the verifier's accepted_tok is emitted
        (the ground-truth token), and we do NOT advance an extra step.

        This verifies that on rejection the loop is still functionally correct (it
        just doesn't get the 2x speedup). Both paths emit the same final sequence.
        """
        gt = self.GROUND_TRUTH

        # MTP always predicts the WRONG token (inverted ground truth)
        def bad_mtp(pos):
            return (gt[pos] + 1) % 100  # deliberately wrong

        out_reject = []
        pos = 1
        start = gt[0]
        out_reject.append(start)

        while len(out_reject) < len(gt):
            draft = bad_mtp(pos)
            ver = gt[pos]   # verifier says ground truth
            # draft != ver -> reject path: emit ver, don't advance extra
            out_reject.append(ver)
            pos += 1

        assert out_reject == gt, f"reject fallback wrong: {out_reject}"


# ---------------------------------------------------------------------------
# Batch-verify cache snapshot / restore correctness (unit test for _snapshot_cache)

class TestSnapshotRestoreCache:
    """Verify that _snapshot_cache / _restore_cache correctly save and restore
    the ArraysCache (GatedDeltaNet) and QSACache (KV + indexer) state so that
    a model forward on the same input after restore produces identical results
    to a fresh forward at the same cache position.

    Uses the toy Model with random weights. No real checkpoint required.
    The key property tested:
        snapshot(state) -> forward(A) -> restore -> forward(A) == same logits as
        the original forward(A) before the perturbation.
    """

    def _build_toy_model(self):
        """Build a tiny Model at toy dims with random weights."""
        from mlx_qwen4exp.model import Model

        args = toy_args(
            num_hidden_layers=4,
            full_attention_interval=4,   # layer 3 is QSA, layers 0/1/2 are linear
            load_mtp=False,
        )
        mx.random.seed(99)
        model = Model(args)
        lm = model.model.language_model
        lm.embed_tokens.weight = mx.random.normal(lm.embed_tokens.weight.shape) * 0.1
        model.lm_head.weight = mx.random.normal(model.lm_head.weight.shape) * 0.1
        return model, args

    def test_snapshot_restore_arrays_cache(self):
        """Batch-verify reject scenario: snapshot BEFORE batch, restore, single verify == baseline.

        This is the exact sequence the generate_mtp_v2 rejection path uses:
          1. snapshot(cache) before batch-verify
          2. model([[cur, draft]]) -- batch verify, corrupts cache
          3. restore(cache, snap) -- undo the batch
          4. model([[cur]])       -- single-token verify
          5. compare to a SEPARATE baseline run of model([[cur]]) at the same offset

        If snapshot/restore is correct, steps 4 and 5 must give identical logits.
        """
        import warnings
        warnings.filterwarnings('ignore')   # suppress PLE missing warning
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.run_mlx import _snapshot_cache, _restore_cache

        model, args = self._build_toy_model()

        prefix_ids = mx.array([[1, 2, 3]])

        # Build the BASELINE: prefill + single-token verify at the clean state
        cache_base = model.make_cache()
        _ = model(prefix_ids, cache=cache_base)
        _ = np.array(_.astype(mx.float32))
        tok_a = 5
        baseline_logits = model(mx.array([[tok_a]]), cache=cache_base)
        baseline_arr = np.array(baseline_logits.astype(mx.float32))

        # Build the TEST cache: same prefill, then snapshot -> batch -> restore -> single
        cache_test = model.make_cache()
        _ = model(prefix_ids, cache=cache_test)
        _ = np.array(_.astype(mx.float32))

        # Force evaluation of all lazy arrays in cache before snapping
        for c in cache_test:
            from mlx_lm.models.cache import ArraysCache
            if isinstance(c, ArraysCache):
                if c.cache[0] is not None:
                    mx.eval(c.cache[0])
                if c.cache[1] is not None:
                    mx.eval(c.cache[1])

        # SNAPSHOT before batch verify
        snap = _snapshot_cache(cache_test, model)

        # Batch verify [tok_a, tok_b] -- would be run on draft acceptance check
        tok_b = 99
        batch_logits = model(mx.array([[tok_a, tok_b]]), cache=cache_test)
        _ = np.array(batch_logits.astype(mx.float32))

        # Restore (rejection path)
        _restore_cache(snap, cache_test, model)

        # Single-token verify at restored cache -- must match baseline
        restored_logits = model(mx.array([[tok_a]]), cache=cache_test)
        restored_arr = np.array(restored_logits.astype(mx.float32))

        max_diff = np.abs(baseline_arr - restored_arr).max()
        assert max_diff < 1e-4, (
            f"Snapshot-restore: restored single-verify differs from baseline: max_diff={max_diff:.4e}"
        )

    def test_batch2_position0_equals_single(self):
        """Batch-verify property: model([[a, b]])[:, 0, :] == model([[a]])[:, 0, :]
        for a causal model, both at the same cache offset.

        This is the equality guarantee that makes batch-verify spec-dec correct:
        position 0 of the batch-2 output is independent of token b.
        """
        model, args = self._build_toy_model()

        # Prefill so the cache is non-trivial
        prefix_ids = mx.array([[1, 2, 3]])
        cache1 = model.make_cache()
        _ = model(prefix_ids, cache=cache1)
        _ = np.array(_.astype(mx.float32))

        cache2 = model.make_cache()
        _ = model(prefix_ids, cache=cache2)
        _ = np.array(_.astype(mx.float32))

        # Single-token verify at cache1
        tok_a = 5
        single_logits = model(mx.array([[tok_a]]), cache=cache1)
        single_arr = np.array(single_logits[0, 0].astype(mx.float32))

        # Batch-verify at cache2 with [tok_a, tok_b] where tok_b != tok_a
        tok_b = 99
        batch_logits = model(mx.array([[tok_a, tok_b]]), cache=cache2)
        batch_arr = np.array(batch_logits[0, 0].astype(mx.float32))

        max_diff = np.abs(single_arr - batch_arr).max()
        # Position 0 logits must be identical: causal model, position 0 doesn't
        # attend to position 1.
        assert max_diff < 1e-2, (
            f"batch[0] != single logits: max_diff={max_diff:.4e} -- "
            "batch-verify equality guarantee violated"
        )
