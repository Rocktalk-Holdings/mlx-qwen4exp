"""End-to-end tests for mlx_qwen4exp.model — the full Qwen3.8-Flash-Next integrator.

Toy config throughout (nothing needs the 355 GB checkpoint):
    hidden 32, hc 4, lowrank 8, 8 layers @ interval 4  => 6 linear + 2 QSA,
    PLE at layer 1 (0-based), 4 experts top-2, moe_inter 16, vocab 100,
    a tiny 200-row n-gram table provided directly.

Forcing rule (harness quirk: a security hook false-positives on the token "eval", so no
mx.eval in tests): every value assertion goes through np.array(...) / float(...), which
forces the MLX graph, so any NaN surfaces there.

Run:
    /opt/homebrew/bin/python3.11 -m pytest tests/test_model.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# repo root on sys.path so `mlx_qwen4exp` (implicit namespace package) imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from mlx_lm.models.cache import ArraysCache  # noqa: E402

from mlx_qwen4exp import ModelArgs, Model  # noqa: E402
from mlx_qwen4exp.model import QSACache  # noqa: E402


# --------------------------------------------------------------------------- fixtures
HIDDEN = 32
HC = 4
LOWRANK = 8
N_LAYERS = 8
VOCAB = 100
EOS = 99
N_ROWS = 200  # tiny n-gram table


def toy_args(**over) -> ModelArgs:
    base = dict(
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=LOWRANK,
        num_hidden_layers=N_LAYERS,
        full_attention_interval=4,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        indexer_n_heads=2,
        indexer_head_dim=8,
        indexer_kv_heads=1,
        indexer_budget=8,
        indexer_compress_ratio=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        linear_num_key_heads=4,
        linear_key_head_dim=8,
        linear_num_value_heads=12,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        ple_layer_ids=[2],  # 1-based => 0-based layer index 1
        ngram_size=3,
        heads_per_ngram=2,
        ple_embed_dim=HIDDEN,
        ple_conv_kernel_size=4,
        eos_token_id=EOS,
    )
    base.update(over)
    return ModelArgs(**base)


def build_model(seed: int = 0) -> Model:
    mx.random.seed(seed)
    return Model(toy_args())


def attach_table(m: Model, seed: int = 1) -> Model:
    """Attach a small PLE table + hash constants so the PLE block runs."""
    args = m.args
    nh = args.ple_n_heads
    mx.random.seed(seed)
    m._ple_table = mx.random.normal((N_ROWS, args.ple_head_dim)) * 0.1
    # multipliers deliberately huge so the uint64 hash overflows/wraps
    m._ple_multipliers = np.array(
        [6364136223846793005, -4232172114205192657, 1181783497276652981],
        dtype=np.int64,
    )
    head_vocab = N_ROWS // nh  # keep every hashed row inside the table
    m._ple_vocab_sizes = np.array([head_vocab] * nh, dtype=np.int64)
    m._ple_offsets = np.array([head_vocab * i for i in range(nh)], dtype=np.int64)
    return m


# --------------------------------------------------------------------------- forward
def test_forward_shape_and_no_nan_with_table():
    m = attach_table(build_model())
    m.reset_ple_state()
    x = mx.array(np.random.randint(0, VOCAB - 1, size=(2, 5)))
    out = m(x)
    assert out.shape == (2, 5, VOCAB)
    a = np.array(out)
    assert not np.isnan(a).any()
    assert np.isfinite(a).all()


def test_forward_degraded_without_table_warns_once():
    m = build_model()  # no table attached
    x = mx.array(np.random.randint(0, VOCAB - 1, size=(1, 4)))
    with pytest.warns(RuntimeWarning):
        out = m(x)
    assert out.shape == (1, 4, VOCAB)
    assert not np.isnan(np.array(out)).any()
    # a second call must NOT warn again (one-shot flag)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        out2 = m(x)
    assert out2.shape == (1, 4, VOCAB)


# --------------------------------------------------------------------------- make_cache
def test_make_cache_types_per_layer():
    m = build_model()
    cache = m.make_cache()
    assert len(cache) == N_LAYERS
    for i, (layer, c) in enumerate(zip(m.layers, cache)):
        if layer.is_linear:
            assert isinstance(c, ArraysCache), f"layer {i} should be ArraysCache"
        else:
            assert isinstance(c, QSACache), f"layer {i} should be QSACache"
    # exactly the (i+1)%4==0 layers are QSA
    qsa = [i for i, l in enumerate(m.layers) if not l.is_linear]
    assert qsa == [3, 7]


# --------------------------------------------------------------------------- incremental == single-shot
def test_incremental_generation_matches_single_shot_with_ple():
    """10-token prefill + 3 single-token decodes == 13-token single-shot on the shared
    positions, WITH the PLE table active and prev-token threading exercised. This is the
    conv/state/prev-token bug detector."""
    m = attach_table(build_model(seed=2), seed=5)
    T = 13
    x = mx.array(np.random.randint(0, VOCAB - 1, size=(1, T)))

    m.reset_ple_state()
    full = np.array(m(x))

    m.reset_ple_state()
    cache = m.make_cache()
    outs = [np.array(m(x[:, :10], cache=cache))]
    for t in range(10, T):
        outs.append(np.array(m(x[:, t : t + 1], cache=cache)))
    chunked = np.concatenate(outs, axis=1)

    assert full.shape == chunked.shape == (1, T, VOCAB)
    assert not np.isnan(chunked).any()
    assert np.allclose(full, chunked, atol=1e-4), (
        f"max abs diff {np.abs(full - chunked).max():.3e}"
    )


def test_incremental_generation_various_chunkings():
    """Feed the same 13 tokens as 1s, as (7,6), and single-shot; all must agree."""
    m = attach_table(build_model(seed=4), seed=6)
    T = 13
    x = mx.array(np.random.randint(0, VOCAB - 1, size=(1, T)))

    m.reset_ple_state()
    full = np.array(m(x))

    # all ones
    m.reset_ple_state()
    cache = m.make_cache()
    outs = [np.array(m(x[:, t : t + 1], cache=cache)) for t in range(T)]
    ones = np.concatenate(outs, axis=1)
    assert np.allclose(full, ones, atol=1e-4), np.abs(full - ones).max()

    # (7, 6)
    m.reset_ple_state()
    cache = m.make_cache()
    o = [np.array(m(x[:, :7], cache=cache)), np.array(m(x[:, 7:], cache=cache))]
    split = np.concatenate(o, axis=1)
    assert np.allclose(full, split, atol=1e-4), np.abs(full - split).max()


# --------------------------------------------------------------------------- sanitize
def _synthetic_hf_weights(args: ModelArgs) -> dict:
    """A synthetic HF-style checkpoint at toy dims, using the REAL key patterns.

    Zero-centred gammas are stored as (target - 1) so the loader's +1 must recover them.
    We store a KNOWN sentinel (target value 3.0) in every zero-centred gamma so the test
    can assert the folding happened (loaded == input + 1 == 3.0).
    Includes visual.* / mtp.* junk and the PLE I64 constants that must be dropped.
    """
    D = args.hidden_size
    hc_dim = args.hc_dim
    lowrank = args.hc_lowrank
    hc = args.hc_count
    HD = args.head_dim
    H = args.num_attention_heads
    KV = args.num_key_value_heads
    E = args.num_experts
    inter = args.moe_intermediate_size
    shared_inter = args.shared_expert_intermediate_size
    idx_h = args.indexer_n_heads
    idx_d = args.indexer_head_dim
    key_dim = args.key_dim
    value_dim = args.value_dim
    conv_dim = args.conv_dim
    nvh = args.linear_num_value_heads
    hvd = args.linear_value_head_dim
    ple_hcdim = hc_dim

    def rnd(*shape):
        return mx.random.normal(shape).astype(mx.float32)

    # sentinel for zero-centred gammas: store (3.0 - 1.0) = 2.0 so loaded == 3.0
    GAMMA_STORED = 2.0
    GAMMA_TARGET = 3.0

    def zc(n):  # zero-centred gamma of length n, stored as target-1
        return mx.full((n,), GAMMA_STORED)

    w = {}
    # global language weights
    w["model.language_model.embed_tokens.weight"] = rnd(args.vocab_size, D)
    w["lm_head.weight"] = rnd(args.vocab_size, D)
    # final mixer (no inject) — hc_norm is zero-centred
    w["model.language_model.hyper_connection_mixer.hc_norm.weight"] = zc(hc_dim)
    w["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"] = rnd(
        lowrank, hc_dim
    )
    w["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"] = rnd(
        hc_dim, lowrank
    )

    for l in range(args.num_hidden_layers):
        P = f"model.language_model.layers.{l}"
        # both hyper-connections
        for hcn in ("attn_hyper_connection", "mlp_hyper_connection"):
            w[f"{P}.{hcn}.hc_norm.weight"] = zc(hc_dim)
            w[f"{P}.{hcn}.input_mix_weight_down.weight"] = rnd(lowrank, hc_dim)
            w[f"{P}.{hcn}.input_mix_weight_up.weight"] = rnd(hc_dim, lowrank)
            w[f"{P}.{hcn}.block_inject_weight.weight"] = rnd(hc, hc_dim)

        # MoE — every layer
        w[f"{P}.mlp.gate.weight"] = rnd(E, D)
        w[f"{P}.mlp.experts.gate_up_proj"] = rnd(E, 2 * inter, D)
        w[f"{P}.mlp.experts.down_proj"] = rnd(E, D, inter)
        w[f"{P}.mlp.shared_expert.gate_proj.weight"] = rnd(shared_inter, D)
        w[f"{P}.mlp.shared_expert.up_proj.weight"] = rnd(shared_inter, D)
        w[f"{P}.mlp.shared_expert.down_proj.weight"] = rnd(D, shared_inter)
        w[f"{P}.mlp.shared_expert_gate.weight"] = rnd(1, D)

        if args.is_linear(l):
            w[f"{P}.linear_attn.in_proj_qkv.weight"] = rnd(conv_dim, D)
            w[f"{P}.linear_attn.in_proj_z.weight"] = rnd(value_dim, D)
            w[f"{P}.linear_attn.in_proj_b.weight"] = rnd(nvh, D)
            w[f"{P}.linear_attn.in_proj_a.weight"] = rnd(nvh, D)
            w[f"{P}.linear_attn.conv1d.weight"] = rnd(conv_dim, 1, args.linear_conv_kernel_dim)
            w[f"{P}.linear_attn.A_log"] = rnd(nvh)
            w[f"{P}.linear_attn.dt_bias"] = rnd(nvh)
            # linear_attn.norm is NOT zero-centred (excluded from +1) -> store a plain value
            w[f"{P}.linear_attn.norm.weight"] = mx.ones((hvd,)) * 5.0
            w[f"{P}.linear_attn.out_proj.weight"] = rnd(D, value_dim)
        else:
            w[f"{P}.self_attn.q_proj.weight"] = rnd(H * 2 * HD, D)
            w[f"{P}.self_attn.k_proj.weight"] = rnd(KV * HD, D)
            w[f"{P}.self_attn.v_proj.weight"] = rnd(KV * HD, D)
            w[f"{P}.self_attn.o_proj.weight"] = rnd(D, H * HD)  # [hidden, H*HD] (SPEC:111)
            # q_norm/k_norm follow the base +1 rule (zero-centred)
            w[f"{P}.self_attn.q_norm.weight"] = zc(HD)
            w[f"{P}.self_attn.k_norm.weight"] = zc(HD)
            # fused indexer qk projection: q rows then k rows
            w[f"{P}.self_attn.indexer.index_qk_proj.weight"] = rnd(idx_h * idx_d + idx_d, D)
            # indexer layernorms zero-centred (override +1)
            w[f"{P}.self_attn.indexer.q_layernorm.weight"] = zc(idx_d)
            w[f"{P}.self_attn.indexer.k_layernorm.weight"] = zc(idx_d)

        if l in args.ple_layers:
            w[f"{P}.ple.key_proj.weight"] = rnd(ple_hcdim, args.ple_embed_dim)
            w[f"{P}.ple.value_proj.weight"] = rnd(D, args.ple_embed_dim)
            # ple norms zero-centred (override +1)
            w[f"{P}.ple.norm_key.weight"] = zc(ple_hcdim)
            w[f"{P}.ple.norm_query.weight"] = zc(ple_hcdim)
            w[f"{P}.ple.norm_conv.weight"] = zc(ple_hcdim)
            w[f"{P}.ple.conv1d.weight"] = rnd(ple_hcdim, 1, args.ple_conv_kernel_size)
            # PLE I64 constants + a couple of ngram shards -> must be DROPPED by sanitize
            w[f"{P}.ple.ple_embedding.layer_multipliers"] = mx.zeros((args.ngram_size,))
            w[f"{P}.ple.ple_embedding.ngram_heads_offsets"] = mx.zeros((args.ple_n_heads,))
            w[f"{P}.ple.ple_embedding.ngram_heads_vocab_sizes"] = mx.zeros((args.ple_n_heads,))
            w[f"{P}.ple.ple_embedding.ngram_embedding.shard_0.weight"] = rnd(10, args.ple_head_dim)
            w[f"{P}.ple.ple_embedding.ngram_embedding.shard_1.weight"] = rnd(10, args.ple_head_dim)

    # junk that must be dropped
    w["model.visual.blocks.0.attn.qkv.weight"] = rnd(4, 4)
    w["model.visual.patch_embed.proj.weight"] = rnd(4, 4)
    w["mtp.layers.0.fc.weight"] = rnd(4, 4)
    w["mtp.norm.weight"] = rnd(4)
    return w, GAMMA_TARGET


def test_sanitize_loads_strict_and_folds_gammas():
    args = toy_args()
    mx.random.seed(0)
    m = Model(args)
    raw, gamma_target = _synthetic_hf_weights(args)

    sanitized = m.sanitize(dict(raw))

    # junk dropped
    for k in sanitized:
        assert not k.startswith("model.visual."), k
        assert not k.startswith("mtp."), k
        assert ".ple.ple_embedding." not in k, k

    # strict load MUST succeed: every produced key maps to a real param, no leftovers
    m.load_weights(list(sanitized.items()), strict=True)

    # -- assert the +1 folding happened on a representative gamma of each rule branch --
    lm = m.model.language_model
    # BASE rule: hc_norm (final mixer)
    hc_g = np.array(lm.hyper_connection_mixer.hc_norm)
    assert np.allclose(hc_g, gamma_target, atol=1e-5), ("hc_norm not folded", hc_g[:3])

    # find a QSA layer and a linear layer
    qsa_layer = next(l for l in m.layers if not l.is_linear)
    lin_layer = next(l for l in m.layers if l.is_linear)
    ple_layer = next(l for l in m.layers if l.has_ple)

    # BASE rule: self_attn.q_norm / k_norm
    assert np.allclose(np.array(qsa_layer.self_attn.q_norm), gamma_target, atol=1e-5)
    assert np.allclose(np.array(qsa_layer.self_attn.k_norm), gamma_target, atol=1e-5)
    # OVERRIDE: indexer q/k layernorms
    assert np.allclose(np.array(qsa_layer.self_attn.indexer.q_layernorm), gamma_target, atol=1e-5)
    assert np.allclose(np.array(qsa_layer.self_attn.indexer.k_layernorm), gamma_target, atol=1e-5)
    # OVERRIDE: ple norms
    assert np.allclose(np.array(ple_layer.ple.norm_key), gamma_target, atol=1e-5)
    assert np.allclose(np.array(ple_layer.ple.norm_query), gamma_target, atol=1e-5)
    assert np.allclose(np.array(ple_layer.ple.norm_conv), gamma_target, atol=1e-5)

    # EXCLUSION: linear_attn.norm.weight must NOT be folded (stored 5.0, loaded 5.0)
    assert np.allclose(np.array(lin_layer.linear_attn.norm.weight), 5.0, atol=1e-5), (
        "linear_attn.norm must NOT get +1"
    )

    # -- expert split produced the SwitchLinear shapes --
    sm = qsa_layer.mlp.switch_mlp
    assert sm.gate_proj.weight.shape == (args.num_experts, args.moe_intermediate_size, args.hidden_size)
    assert sm.up_proj.weight.shape == (args.num_experts, args.moe_intermediate_size, args.hidden_size)
    assert sm.down_proj.weight.shape == (args.num_experts, args.hidden_size, args.moe_intermediate_size)

    # -- indexer split produced q/k proj shapes --
    iq = qsa_layer.self_attn.indexer.index_q_proj.weight
    ik = qsa_layer.self_attn.indexer.index_k_proj.weight
    assert iq.shape == (args.indexer_n_heads * args.indexer_head_dim, args.hidden_size)
    assert ik.shape == (args.indexer_head_dim, args.hidden_size)


def test_sanitize_expert_split_values_gate_first():
    """The gate half must be the FIRST moe_inter rows of the fused tensor (gate||up)."""
    args = toy_args()
    mx.random.seed(0)
    m = Model(args)
    raw, _ = _synthetic_hf_weights(args)
    # grab a known fused tensor before sanitize
    key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    fused = np.array(raw[key])
    inter = args.moe_intermediate_size
    sanitized = m.sanitize(dict(raw))
    gate = np.array(sanitized["model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight"])
    up = np.array(sanitized["model.language_model.layers.0.mlp.switch_mlp.up_proj.weight"])
    assert np.array_equal(gate, fused[:, :inter, :])
    assert np.array_equal(up, fused[:, inter:, :])


def test_sanitize_indexer_split_rows():
    """index_qk_proj rows 0:n_q -> q, rows n_q: -> k."""
    args = toy_args()
    mx.random.seed(0)
    m = Model(args)
    raw, _ = _synthetic_hf_weights(args)
    P = "model.language_model.layers.3.self_attn"  # layer 3 is QSA
    key = f"{P}.indexer.index_qk_proj.weight"
    assert key in raw
    src = np.array(raw[key])
    n_q = args.indexer_n_heads * args.indexer_head_dim
    sanitized = m.sanitize(dict(raw))
    q = np.array(sanitized[f"{P}.indexer.index_q_proj.weight"])
    k = np.array(sanitized[f"{P}.indexer.index_k_proj.weight"])
    assert np.array_equal(q, src[:n_q])
    assert np.array_equal(k, src[n_q:])


def test_sanitize_conv1d_shapes():
    """linear_attn conv1d -> [C,K,1] (moveaxis) so nn.Conv1d loads; ple conv1d -> [C,K]."""
    args = toy_args()
    mx.random.seed(0)
    m = Model(args)
    raw, _ = _synthetic_hf_weights(args)
    sanitized = m.sanitize(dict(raw))
    # linear layer 0
    lin_conv = sanitized["model.language_model.layers.0.linear_attn.conv1d.weight"]
    # nn.Conv1d(groups=C) weight is [C_out, K, C_in/groups] = [conv_dim, K, 1]
    assert lin_conv.shape == (args.conv_dim, args.linear_conv_kernel_dim, 1)
    # ple layer: bare mx.array `conv1d` (".weight" dropped), shape [C, K]
    ple_conv = sanitized["model.language_model.layers.1.ple.conv1d"]
    assert ple_conv.shape == (args.hc_dim, args.ple_conv_kernel_size)
    assert "model.language_model.layers.1.ple.conv1d.weight" not in sanitized


# --------------------------------------------------------------------------- 48-layer toy forward (no NaN)
def test_full_depth_48_layer_toy_forward_no_nan():
    """A forward pass through 48 layers at TOY dims (the real depth, the real layer_types
    pattern, the real PLE placement) confirms the whole stack is finite through every
    block type at full depth (SPEC 6.3) — without the 355 GB checkpoint."""
    args = toy_args(num_hidden_layers=48)  # 48 layers, interval 4 => 36 linear + 12 QSA
    m = attach_table(Model(args), seed=8)
    m.reset_ple_state()
    x = mx.array(np.random.randint(0, VOCAB - 1, size=(2, 6)))
    out = m(x)
    a = np.array(out)
    assert out.shape == (2, 6, VOCAB)
    assert not np.isnan(a).any()
    assert np.isfinite(a).all()
    # QSA layers land exactly at (i+1)%4==0
    qsa = [i for i, l in enumerate(m.layers) if not l.is_linear]
    assert qsa == [i for i in range(48) if (i + 1) % 4 == 0]
