"""Gated DeltaNet (linear-attention) layer for qwen4_exp (Qwen3.8-Flash-Next).

This is the same delta rule as Qwen3-Next (mlx_lm.models.qwen3_next) with three
concrete deviations, all pinned by SPEC 4.4 and the C++ ground truth
(llama.cpp/src/models/qwen4exp.cpp:361-387, 693-819):

  1. FOUR SEPARATE projections instead of the fused ``in_proj_qkvz`` / ``in_proj_ba``:
     ``in_proj_qkv`` (q|k|v on the output axis), ``in_proj_z``, ``in_proj_b``, ``in_proj_a``.
     So ``fix_query_key_value_ordering`` is NOT reused -- the ordering is trivial here
     (the checkpoint already lays q|k|v out contiguously on the output axis).

  2. q,k are L2-normalised (``ggml_l2_norm``, cpp:787-788), NOT the scaled rms_norm the
     qwen3_next path uses. eps = rms_norm_eps.

  3. The output gate is SIGMOID, not silu (cpp:377-387, "the one numerical difference
     from Qwen3.5's GDN"). Implemented in ``Qwen4ExpRMSNormGated`` below:
         out = rms_norm(x, weight, eps) * sigmoid(gate)

The delta-rule kernel itself (``gated_delta_update``) is REUSED verbatim from
``mlx_lm.models.gated_delta``. It takes RAW ``a`` and ``b`` and internally computes
``beta = sigmoid(b)`` and ``g = exp(-exp(A_log) * softplus(a + dt_bias))`` via
``compute_g`` -- so this module must pass a,b raw and NOT pre-apply sigmoid/softplus.
It also handles the 16-key-head vs 48-value-head broadcast internally
(``mx.repeat(q,k, Hv//Hk, -2)`` in the ops path; ``hk_idx = hv_idx/(Hv/Hk)`` in the
Metal kernel path), so q,k are passed with 16 heads and v with 48.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gated_delta import gated_delta_update

from .config import ModelArgs


@partial(mx.compile, shapeless=True)
def _rms_norm_sigmoid_gate(x: mx.array, weight: mx.array, gate: mx.array, eps: float):
    """out = rms_norm(x, weight, eps) * sigmoid(gate), in float32 then cast back.

    Mirrors ``build_norm_gated`` (cpp:377-387): normalize FIRST, sigmoid the gate,
    multiply. The rms_norm reduces over the last axis (head_v_dim) with ``weight``
    applied inside, exactly like a standard RMSNorm.
    """
    xn = mx.fast.rms_norm(x, weight, eps)
    g = mx.sigmoid(gate.astype(mx.float32))
    return (xn.astype(mx.float32) * g).astype(x.dtype)


class Qwen4ExpRMSNormGated(nn.Module):
    """RMSNorm with a SIGMOID output gate (the qwen4_exp deviation from qwen3_next).

    qwen3_next's ``Qwen3NextRMSNormGated`` gates with silu (swiglu). This one gates
    with sigmoid. Do not "fix" it back to silu -- the sigmoid choice is pinned by a
    behaviour test (see tests/test_deltanet.py::test_sigmoid_gate_not_silu).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        if gate is None:
            return mx.fast.rms_norm(hidden_states, self.weight, self.eps).astype(
                hidden_states.dtype
            )
        return _rms_norm_sigmoid_gate(hidden_states, self.weight, gate, self.eps)


class GatedDeltaNet(nn.Module):
    """Linear-attention layer. Parameter attribute names match the checkpoint with the
    ``{L}.linear_attn.`` prefix stripped."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_k_heads = args.linear_num_key_heads          # 16
        self.num_v_heads = args.linear_num_value_heads        # 48
        self.head_k_dim = args.linear_key_head_dim            # 128
        self.head_v_dim = args.linear_value_head_dim          # 128
        self.key_dim = args.key_dim                           # 2048
        self.value_dim = args.value_dim                       # 6144
        self.conv_dim = args.conv_dim                         # 10240 = 2*2048 + 6144
        self.conv_kernel_size = args.linear_conv_kernel_dim   # 4
        self.eps = args.rms_norm_eps
        # The qwen4_exp GATED_DELTA_NET op scales its readout by 1/sqrt(head_v_dim)
        # (ops.cpp:10824, `attn_data[j] = sum * scale`, scale = 1/sqrt(S_v)). mlx_lm's
        # gated_delta_update omits this, so we re-apply it on the delta output below.
        self.delta_out_scale = 1.0 / (self.head_v_dim ** 0.5)

        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by "
                f"num_k_heads ({self.num_k_heads})"
            )

        # Four separate input projections (NOT fused like qwen3_next).
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)  # 10240
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)   # 6144
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # 48
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # 48

        # Depthwise causal conv over time, kernel 4. Same attribute name and weight
        # layout as qwen3_next so the parent Model.sanitize's ``conv1d.weight`` moveaxis
        # transform ([C,1,K] -> squeeze -> moveaxis to [C_out,K,C_in]) applies unchanged.
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.A_log = mx.zeros((self.num_v_heads,))
        self.dt_bias = mx.zeros((self.num_v_heads,))

        # Gated output norm over head_v_dim (128), sigmoid gate.
        self.norm = Qwen4ExpRMSNormGated(self.head_v_dim, eps=self.eps)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)  # 6144 -> 2560

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape

        qkv = self.in_proj_qkv(x)                # [B,S,10240]  q|k|v contiguous on last axis
        z = self.in_proj_z(x)                    # [B,S,6144]
        b = self.in_proj_b(x)                    # [B,S,48]  RAW
        a = self.in_proj_a(x)                    # [B,S,48]  RAW

        # ---- causal depthwise conv over time, state threaded through the cache ----
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        conv_input = mx.concatenate([conv_state, qkv], axis=1)  # [B, S+K-1, conv_dim]

        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = conv_input[:, -n_keep:, :]

        conv_out = nn.silu(self.conv1d(conv_input))  # [B,S,conv_dim], causal (no padding)

        # split q[2048] | k[2048] | v[6144] then reshape into heads
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        # L2-normalise q,k over the last (head) axis (ggml_l2_norm, cpp:787-788).
        q = _l2_normalize(q, self.eps)
        k = _l2_normalize(k, self.eps)

        state = cache[1] if cache is not None else None

        # gated_delta_update takes RAW a,b and handles the 16->48 head broadcast itself.
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(S)

        # qwen4_exp scales the delta readout by 1/sqrt(head_v_dim); mlx_lm's kernel does
        # not, so apply it here before the gated norm (ggml GATED_DELTA_NET, ops.cpp:10824).
        out = out * self.delta_out_scale

        # gated output norm with SIGMOID gate; z reshaped to [B,S,48,128].
        out = self.norm(out, z.reshape(B, S, self.num_v_heads, self.head_v_dim))
        return self.out_proj(out.reshape(B, S, self.value_dim))


def _l2_normalize(x: mx.array, eps: float) -> mx.array:
    """L2-normalise over the last axis, matching ``ggml_l2_norm``:
    x / sqrt(sum(x^2) + eps). Computed in float32, cast back."""
    xf = x.astype(mx.float32)
    denom = mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * denom).astype(x.dtype)
