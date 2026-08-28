"""Hyper-connections (gated residual machinery) for qwen4_exp / Qwen3.8-Flash-Next.

Qwen3.8-Flash-Next has no layer norms. Instead it carries a WIDE residual stream of
shape ``[B, T, hc, D]`` (``hc = hc_count = 4``, ``D = hidden_size = 2560``) and, at every
block site, does:

    cur, inject = HyperConnection(res_wide)        # mixes hc streams down to [B, T, D]
    cur         = block(cur)                       # attention / deltanet / moe
    res_wide    = hc_combine(res_wide, cur, inject, hc)

The final ``HyperConnection`` (``model.language_model.hyper_connection_mixer``, built with
``with_inject=False``) doubles as the output norm before ``lm_head``.

Reference implementation (ground truth): ``llama.cpp/src/models/qwen4exp.cpp``
  - ``build_hc_mix``      lines 197-243
  - ``build_hc_combine``  lines 246-266
  - wide-residual init    lines 295-299

See SPEC.md section 4.1.

Weight-name contract
--------------------
Attribute names here are chosen so that a checkpoint key, after stripping its module
prefix, lands directly on the right parameter::

    {L}.attn_hyper_connection.hc_norm.weight                -> hc_norm          [10240]
    {L}.attn_hyper_connection.input_mix_weight_down.weight  -> ...down.weight   [320, 10240]
    {L}.attn_hyper_connection.input_mix_weight_up.weight    -> ...up.weight     [10240, 320]
    {L}.attn_hyper_connection.block_inject_weight.weight    -> ...inject.weight [4, 10240]

NOTE ``hc_norm`` is a bare ``mx.array`` parameter, not an ``nn.Module``, so the
checkpoint's ``hc_norm.weight`` suffix must be rewritten to plain ``hc_norm`` in
``sanitize`` (``HyperConnection.sanitize_prefix`` does this for a weight dict).

NOTE also that the converter folds these gammas to ``1 + w`` (qwen4exp.cpp:211). That
folding is a *load-time* concern and is deliberately NOT done here -- this module
consumes gammas that are already in their final form.
"""

from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs

__all__ = ["HyperConnection", "hc_combine", "hc_init", "grouped_rms_norm"]


# --------------------------------------------------------------------------- helpers
def grouped_rms_norm(
    x_wide: mx.array,
    weight: Optional[mx.array],
    eps: float,
) -> mx.array:
    """Grouped RMSNorm: reduce over D, scale over the flattened ``hc * D``.

    This is *not* a plain RMSNorm. ``x_wide`` is ``[..., hc, D]``; the mean-square is
    taken per-stream over the last axis (``D``) only, and then the ``[hc * D]`` gamma is
    applied to the flattened view. Returns the FLATTENED ``[..., hc * D]`` array, which
    is the layout the low-rank mix and the inject projection both want.

    ``weight=None`` skips the scaling (returns the flattened normalised view).

    Mirrors ``ggml_rms_norm`` on a ``[n_embd, hc, T]`` tensor followed by
    ``ggml_reshape_2d`` + ``ggml_mul`` (qwen4exp.cpp:210-213).
    """
    if x_wide.ndim < 2:
        raise ValueError(
            f"grouped_rms_norm expects at least a [hc, D] array, got shape {x_wide.shape}"
        )
    hc, d = x_wide.shape[-2], x_wide.shape[-1]

    # mx.fast.rms_norm accepts weight=None in mlx >= 0.25 (verified on 0.30.6). Fall back
    # to an explicit float32 computation on older builds rather than silently changing
    # the numerics.
    try:
        xn = mx.fast.rms_norm(x_wide, None, eps)
    except (TypeError, ValueError):
        f32 = x_wide.astype(mx.float32)
        xn = f32 * mx.rsqrt(mx.mean(f32 * f32, axis=-1, keepdims=True) + eps)
        xn = xn.astype(x_wide.dtype)

    xn = xn.reshape(*xn.shape[:-2], hc * d)
    if weight is not None:
        if weight.shape != (hc * d,):
            raise ValueError(
                f"grouped_rms_norm gamma has shape {weight.shape}, expected {(hc * d,)}"
            )
        xn = xn * weight.astype(xn.dtype)
    return xn


def hc_init(embeddings: mx.array, hc_count: int) -> mx.array:
    """Widen ``[B, T, D]`` token embeddings into the ``[B, T, hc, D]`` residual stream.

    The wide residual starts as ``hc`` identical copies of the embedding
    (qwen4exp.cpp:295-299). The result is materialised via ``mx.contiguous`` so that the
    caller holds a real array, not a zero-stride broadcast view -- later
    ``res = res + ...`` updates on a broadcast view would still be correct in MLX, but a
    materialised array keeps memory behaviour predictable and matches ``ggml_repeat_4d``.
    """
    if embeddings.ndim != 3:
        raise ValueError(
            f"hc_init expects [B, T, D] embeddings, got shape {embeddings.shape}"
        )
    if hc_count < 1:
        raise ValueError(f"hc_count must be >= 1, got {hc_count}")
    b, t, d = embeddings.shape
    wide = mx.broadcast_to(embeddings[:, :, None, :], (b, t, hc_count, d))
    return mx.contiguous(wide)


def hc_combine(
    residual_wide: mx.array,
    block_out: mx.array,
    inject: mx.array,
    hc_count: int,
) -> mx.array:
    """Scatter a block's ``[B, T, D]`` output back into the wide residual.

    ``w = 2 * sigmoid(inject / hc)`` is centred on 1, so a zero injection makes this an
    exact plain residual add into every stream (qwen4exp.cpp:254-262).

    Args:
        residual_wide: ``[B, T, hc, D]``
        block_out:     ``[B, T, D]``
        inject:        ``[B, T, hc]``
        hc_count:      number of parallel residual streams

    Returns:
        ``[B, T, hc, D]``
    """
    if residual_wide.ndim != 4:
        raise ValueError(
            f"hc_combine expects residual_wide [B, T, hc, D], got {residual_wide.shape}"
        )
    if block_out.ndim != 3:
        raise ValueError(
            f"hc_combine expects block_out [B, T, D], got {block_out.shape}"
        )
    if inject.ndim != 3:
        raise ValueError(f"hc_combine expects inject [B, T, hc], got {inject.shape}")
    b, t, hc, d = residual_wide.shape
    if hc != hc_count:
        raise ValueError(
            f"residual_wide has hc={hc} but hc_count={hc_count} was passed"
        )
    if block_out.shape != (b, t, d):
        raise ValueError(
            f"block_out shape {block_out.shape} does not match residual "
            f"{residual_wide.shape} (expected {(b, t, d)})"
        )
    if inject.shape != (b, t, hc):
        raise ValueError(
            f"inject shape {inject.shape} does not match residual "
            f"{residual_wide.shape} (expected {(b, t, hc)})"
        )
    if hc_count < 1:
        raise ValueError(f"hc_count must be >= 1, got {hc_count}")

    w = 2.0 * mx.sigmoid(inject / hc_count)  # [B, T, hc]
    return residual_wide + block_out[:, :, None, :] * w[:, :, :, None]


# --------------------------------------------------------------------------- module
class HyperConnection(nn.Module):
    """One hyper-connection site (attention block, mlp block, or the final mixer).

    Args:
        args: the model config; uses ``hidden_size``, ``hc_count``, ``hc_lowrank`` and
            ``rms_norm_eps``.
        with_inject: build ``block_inject_weight``. ``False`` for the terminal
            ``hyper_connection_mixer``, which has no inject tensor in the checkpoint and
            serves purely as the output norm.

    Call:
        ``x_wide [B, T, hc, D] -> (mixed [B, T, D], inject [B, T, hc] or None)``
    """

    def __init__(self, args: ModelArgs, with_inject: bool = True):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        self.hc_lowrank = args.hc_lowrank
        self.hc_dim = args.hc_dim  # hc_count * hidden_size
        self.eps = args.rms_norm_eps
        self.with_inject = with_inject

        if self.hc_count < 1:
            raise ValueError(f"hc_count must be >= 1, got {self.hc_count}")
        if self.hc_lowrank < 1:
            raise ValueError(f"hc_lowrank must be >= 1, got {self.hc_lowrank}")

        # grouped-norm gamma over the flattened [hc * D] layout
        self.hc_norm = mx.ones((self.hc_dim,))

        # low-rank gate:  [hc_dim] -> [hc_lowrank] -> [hc_dim]
        self.input_mix_weight_down = nn.Linear(self.hc_dim, self.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(self.hc_lowrank, self.hc_dim, bias=False)

        if with_inject:
            self.block_inject_weight = nn.Linear(self.hc_dim, self.hc_count, bias=False)

    def __call__(self, x_wide: mx.array) -> Tuple[mx.array, Optional[mx.array]]:
        if x_wide.ndim != 4:
            raise ValueError(
                f"HyperConnection expects [B, T, hc, D], got shape {x_wide.shape}"
            )
        b, t, hc, d = x_wide.shape
        if hc != self.hc_count or d != self.hidden_size:
            raise ValueError(
                f"HyperConnection got [.., hc={hc}, D={d}], "
                f"expected [.., hc={self.hc_count}, D={self.hidden_size}]"
            )

        # grouped RMSNorm: reduce over D, scale over the flattened hc*D -> [B, T, hc_dim]
        xn = grouped_rms_norm(x_wide, self.hc_norm, self.eps)

        # NOTE the 1/hc scale lands on the DOWN-projection output, BEFORE the silu
        # (ggml_scale then ggml_silu, qwen4exp.cpp:218).
        lo = nn.silu(self.input_mix_weight_down(xn) / self.hc_count)  # [B, T, lowrank]
        gate = mx.sigmoid(self.input_mix_weight_up(lo))               # [B, T, hc_dim]

        gated = (xn * gate).reshape(b, t, hc, d)
        mixed = gated.mean(axis=2)                                    # [B, T, D]

        inject = self.block_inject_weight(xn) if self.with_inject else None
        return mixed, inject

    # ------------------------------------------------------------------ loading
    @staticmethod
    def sanitize_prefix(weights: Dict[str, mx.array], prefix: str) -> Dict[str, mx.array]:
        """Rewrite ``{prefix}.hc_norm.weight`` -> ``{prefix}.hc_norm``.

        ``hc_norm`` is a bare parameter here, not an ``nn.Module``, so the checkpoint's
        extra ``.weight`` suffix has to be dropped or ``update()`` will not find it.
        Every other key under ``prefix`` is passed through untouched. Returns a NEW dict;
        the input is not mutated.
        """
        src = f"{prefix}.hc_norm.weight"
        dst = f"{prefix}.hc_norm"
        if src not in weights:
            return dict(weights)
        out = dict(weights)
        out[dst] = out.pop(src)
        return out
