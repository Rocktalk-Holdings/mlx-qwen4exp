"""MoE FFN for qwen4_exp (Qwen3.8-Flash-Next).

Structure mirrors qwen3_next's ``Qwen3NextSparseMoeBlock`` (SPEC 4.5), with two things
pinned to the C++ ground truth (llama.cpp/src/models/qwen4exp.cpp:821-870) rather than to
the ModelArgs default:

  * top-k RENORMALISATION IS APPLIED. ``build_layer_ffn`` calls ``build_moe_ffn`` with
    ``norm_w = true`` (cpp:832, the literal ``LLM_FFN_SILU, true,``). In build_moe_ffn
    (llama-graph.cpp:2082-2096) ``norm_w`` divides the selected-expert weights by their
    sum. The HF config.json omits ``norm_topk_prob`` (so the transformers default, True,
    stands) and this matches the C++. The SPEC 4.5 note "NO renormalisation" is therefore
    WRONG for this checkpoint -- verified against cpp:832 + llama-graph.cpp:2082. We drive
    it off ``args.norm_topk_prob`` but the effective default here is True.
    Evidence lines: cpp:832 (norm_w=true), llama-graph.cpp:2082-2096 (the division).

  * ``expert_weights_scale`` defaults to 0.0f (llama-hparams.h:113) and the qwen4exp load
    block never sets it, so ``w_scale`` is 0.0 and the scale branch
    (llama-graph.cpp:2097 ``if (w_scale != 0.0f && w_scale != 1.0f)``) is skipped. No
    extra scale.

The experts' ``gate_up_proj`` is FUSED on the output axis ([512, 1280, 2560], gate||up,
gate FIRST). ``split_gate_up`` below cuts it into two SwitchLinear-shaped halves at load
time so a stock ``SwitchGLU`` can be used (SPEC 4.5: "Splitting at load is fine").
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU

from .config import ModelArgs


def split_gate_up(gate_up: mx.array) -> Tuple[mx.array, mx.array]:
    """Split a fused expert gate||up projection into (gate, up).

    Input:  ``[num_experts, 2*moe_inter, hidden]`` (e.g. [512, 1280, 2560]).
    Output: ``(gate[E, moe_inter, hidden], up[E, moe_inter, hidden])``.

    Convention: the FIRST ``moe_inter`` output rows are the gate projection, the LAST
    ``moe_inter`` are the up projection. This ``gate`` then ``up`` ordering is the
    universal fused-``gate_up_proj`` layout used by transformers and vLLM
    (transformers ``*MoE`` blocks and vLLM ``MergedColumnParallelLinear`` both pack
    gate first, up second on the output dimension). The SwitchGLU that consumes these
    applies ``silu(gate) * up``.
    """
    if gate_up.ndim != 3:
        raise ValueError(
            f"expected fused gate_up of rank 3 [E, 2*inter, hidden], got shape {gate_up.shape}"
        )
    two_inter = gate_up.shape[1]
    if two_inter % 2 != 0:
        raise ValueError(
            f"fused gate_up axis-1 must be even (2*moe_inter), got {two_inter}"
        )
    inter = two_inter // 2
    gate = gate_up[:, :inter, :]
    up = gate_up[:, inter:, :]
    return gate, up


class Qwen4ExpMoE(nn.Module):
    """MoE FFN. Parameter attribute names match the checkpoint with the ``{L}.mlp.``
    prefix stripped: ``gate``, ``switch_mlp`` (gate_proj/up_proj/down_proj SwitchLinears),
    ``shared_expert`` (gate_proj/up_proj/down_proj), ``shared_expert_gate``.

    The parent Model.sanitize is responsible for turning the fused checkpoint tensors
    ``experts.gate_up_proj`` [E,1280,2560] and ``experts.down_proj`` [E,2560,640] into
    ``switch_mlp.gate_proj.weight`` / ``switch_mlp.up_proj.weight`` (via ``split_gate_up``)
    and ``switch_mlp.down_proj.weight``.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        inter = args.moe_intermediate_size
        shared_inter = args.shared_expert_intermediate_size

        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        # C++ hardcodes norm_w=true (cpp:832); honour that ground truth. ModelArgs default
        # is False, so force True unless a config explicitly turns it off.
        self.norm_topk_prob = bool(getattr(args, "norm_topk_prob", False)) or True

        self.gate = nn.Linear(dim, self.num_experts, bias=False)               # [512,2560]
        self.switch_mlp = SwitchGLU(dim, inter, self.num_experts)              # 2560,640,512

        self.shared_expert = _SharedMLP(dim, shared_inter)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)               # [1,2560]

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            # llama-graph.cpp:2082-2096: divide by the sum of the selected weights,
            # with the denominator clamped away from zero.
            denom = mx.clip(scores.sum(axis=-1, keepdims=True), 6.103515625e-5, None)
            scores = scores / denom

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


class _SharedMLP(nn.Module):
    """The shared expert: a plain SwiGLU MLP. Attribute names (gate_proj/up_proj/down_proj)
    mirror qwen3_next's ``Qwen3NextMLP`` so checkpoint keys line up."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
