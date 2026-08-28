"""MTP (Multi-Token Prediction) draft head for Qwen3.8-Flash-Next.

The checkpoint ships a single ``mtp.*`` namespace that is dropped by default in
``Model.sanitize``. When ``ModelArgs.load_mtp=True`` this module is loaded instead.

Architecture (derived from tensor shapes in ref/tensor_shapes.json + SPEC
notes/MTP-SPEC-2026-08-28.md):

  Input at decode step t:
    last_hidden_wide : [B, 1, hc, D]  — main model's final hc_combine output (the
                       wide residual BEFORE hyper_connection_mixer collapses it).
    next_input_id    : [B, 1] int     — the token whose position we are drafting;
                       i.e. the current accepted token that will be fed back as input.

  Step 1  grouped_rms_norm(last_hidden_wide, gamma=pre_fc_norm_hidden) -> [B,1,hc_dim]
          then fc_hidden projection -> [B,1,D]
          Provenance: pre_fc_norm_hidden [10240]=hc_dim; hc_dim = hc_count*hidden_size.

  Step 2  embed_tokens(next_input_id) -> [B,1,D]  (shared, no +1 gamma)
          rms_norm(e, pre_fc_norm_embedding) -> [B,1,D]
          then fc_embedding projection -> [B,1,D]
          Provenance: pre_fc_norm_embedding [2560]=hidden_size, standard RMSNorm.

  Step 3  fused = h_proj + e_proj   [B,1,D]
          INFERRED from shape (two [2560,2560] matrices, not one [2560,5120]).
          Detection: near-zero acceptance rate or equality-test failure if wrong.

  Step 4  res = hc_init(fused, hc_count)  [B,1,hc,D]

  Step 5  Run mtp.layers.0 (full_attention):
            cur, inj = attn_hyper_connection(res)
            cur = self_attn(cur, cache, idx_cache, pos_offset)
            res = hc_combine(res, cur, inj, hc_count)
            cur, inj = mlp_hyper_connection(res)
            cur = mlp(cur)
            res = hc_combine(res, cur, inj, hc_count)

  Step 6  out, _ = hyper_connection_mixer(res)  [B,1,D]

  Step 7  logits = lm_head(out)   [B,1,vocab]  — shared lm_head from main model.

Weight conventions (notes/MTP-SPEC-2026-08-28.md section 2.5, corrected 2026-08-28):
  - hc_norm gammas (mtp.hyper_connection_mixer.hc_norm, mtp.layers.0.*.hc_norm,
    self_attn.q_norm, k_norm, indexer.q/k_layernorm): drop .weight suffix, +1 fold.
  - pre_fc_norm_hidden, pre_fc_norm_embedding: bare mx.array gammas, WITH +1 fold.
    CORRECTION from SPEC 2.5: originally marked NO +1 (pattern mismatch). Empirical
    evidence shows checkpoint stores (gamma - 1): mean=-0.76 for pre_fc_norm_embedding
    is nonsensical as a raw scale; adding 1 gives mean=0.24, a valid low gamma.
    The (gamma-1) encoding is the HF convention for zero-centred norm weights.
  - fc_hidden, fc_embedding: plain Linear, no +1.
  - indexer.index_qk_proj [640,2560]: split q [512,2560] / k [128,2560].
  - experts.gate_up_proj: split via split_gate_up.

Attribute names mirror the checkpoint keys with the ``mtp.`` prefix stripped so
``load_weights`` maps directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .attention import QSAAttention, IndexerCache, QSACache
from .hyper import HyperConnection, grouped_rms_norm, hc_init, hc_combine
from .moe import Qwen4ExpMoE, split_gate_up

__all__ = ["MTPDraft", "MTPDraftLayer", "make_mtp_cache"]


# --------------------------------------------------------------------------- MTP decoder layer
class MTPDraftLayer(nn.Module):
    """A single full_attention decoder layer for the MTP draft head.

    Attribute names mirror ``mtp.layers.0.*`` checkpoint keys (with the
    ``mtp.layers.0.`` prefix stripped), so ``load_weights`` maps directly.
    The layer is always full_attention (the checkpoint has no linear_attn.* keys).
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        # both hyper-connections present (checkpoint confirms both)
        self.attn_hyper_connection = HyperConnection(args, with_inject=True)
        self.mlp_hyper_connection = HyperConnection(args, with_inject=True)
        # always QSA (mtp.layers.0 config: layer_types=["full_attention"])
        self.self_attn = QSAAttention(args)
        self.mlp = Qwen4ExpMoE(args)

    def __call__(
        self,
        res_wide: mx.array,
        cache: Optional[QSACache],
        pos_offset: int,
    ) -> mx.array:
        """One full-attention layer forward on the wide residual.

        Args:
            res_wide:   [B, T, hc, D]
            cache:      QSACache for this MTP layer (or None for no-cache run)
            pos_offset: number of already-accepted tokens (the current KVCache.offset)

        Returns:
            updated res_wide [B, T, hc, D]
        """
        hc = res_wide.shape[2]

        cur, inj = self.attn_hyper_connection(res_wide)
        kv = cache.kv if cache is not None else None
        idx = cache.idx if cache is not None else None
        cur = self.self_attn(cur, cache=kv, idx_cache=idx, pos_offset=pos_offset)
        res_wide = hc_combine(res_wide, cur, inj, hc)

        cur, inj = self.mlp_hyper_connection(res_wide)
        cur = self.mlp(cur)
        res_wide = hc_combine(res_wide, cur, inj, hc)
        return res_wide


# --------------------------------------------------------------------------- MTPDraft
class MTPDraft(nn.Module):
    """The MTP draft head.

    Attribute names map to ``mtp.*`` checkpoint keys. The shared embed_tokens and
    lm_head are NOT attributes here; they are passed in at call time from the parent
    Model to avoid duplicating the 51.2B embedding table.

    Attribute tree (matches checkpoint namespace ``mtp.{...}``):
        pre_fc_norm_hidden      bare mx.array [hc_dim]
        pre_fc_norm_embedding   bare mx.array [hidden_size]
        fc_hidden               nn.Linear [hidden_size, hidden_size]
        fc_embedding            nn.Linear [hidden_size, hidden_size]
        hyper_connection_mixer  HyperConnection(with_inject=False)
        layers                  list of one MTPDraftLayer
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        D = args.hidden_size
        hc_dim = args.hc_dim

        # bare gamma arrays (no +1 — NOT covered by qwen.py:394 rule, see SPEC 2.5)
        self.pre_fc_norm_hidden = mx.ones((hc_dim,))
        self.pre_fc_norm_embedding = mx.ones((D,))

        # projections from each normalised branch into hidden space
        self.fc_hidden = nn.Linear(D, D, bias=False)
        self.fc_embedding = nn.Linear(D, D, bias=False)

        # the MTP output norm (no inject — mirrors hyper_connection_mixer in main model)
        self.hyper_connection_mixer = HyperConnection(args, with_inject=False)

        # single decoder layer (mtp.layers.0)
        self.layers: List[MTPDraftLayer] = [MTPDraftLayer(args)]

        # store dims for the forward
        self._hc = args.hc_count
        self._eps = args.rms_norm_eps

    # ------------------------------------------------------------------ forward
    def __call__(
        self,
        last_hidden_wide: mx.array,
        next_input_id: mx.array,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
        cache: Optional[QSACache],
        pos_offset: int,
    ) -> mx.array:
        """Draft logits for the next position.

        Args:
            last_hidden_wide: [B, 1, hc, D] — main model's final hc_combine output.
            next_input_id:    [B, 1] int    — the token at position t (already accepted).
            embed_tokens:     shared nn.Embedding from main model.
            lm_head:          shared nn.Linear from main model.
            cache:            QSACache for this layer (may be None).
            pos_offset:       number of accepted tokens = KVCache.offset before this step.

        Returns:
            logits [B, 1, vocab_size]
        """
        if last_hidden_wide.ndim != 4:
            raise ValueError(
                f"MTPDraft expects last_hidden_wide [B, T, hc, D], "
                f"got {last_hidden_wide.shape}"
            )
        if next_input_id.ndim != 2:
            raise ValueError(
                f"MTPDraft expects next_input_id [B, T] int, got {next_input_id.shape}"
            )

        # Step 1: hidden branch — grouped norm over wide residual, then fc_hidden
        # grouped_rms_norm returns [B, T, hc_dim] with gamma=None, then scale
        h_flat = grouped_rms_norm(last_hidden_wide, None, self._eps)   # [B, 1, hc_dim]
        h_flat = h_flat * self.pre_fc_norm_hidden.astype(h_flat.dtype)
        # fc_hidden: [D, D] maps hc_dim... wait: fc_hidden is [D,D] but h_flat is [hc_dim]
        # The mixer step collapses hc_dim->D first. See SPEC 2.2 step 1 note:
        # fc_hidden is [2560, 2560] — it operates on D=2560, NOT hc_dim=10240.
        # So we must first apply the mixer to get [B,T,D], THEN the norm + fc_hidden.
        # Correction from shape algebra: pre_fc_norm_hidden normalizes [hc_dim] BEFORE
        # the mixer collapses it. The fc_hidden [2560, 2560] receives the mixer output.
        # Revised order (consistent with all shapes):
        #   h_flat [B,1,hc_dim] = grouped_rms_norm(last_hidden_wide)
        #   h_norm = h_flat * pre_fc_norm_hidden
        #   h_mixed = hyper_connection_mixer's internal collapse of h_norm
        #             = (h_norm * gate).reshape(...,hc,D).mean(-2) but we want the
        #             pre-gate mixed view, not the full HyperConnection forward.
        # HOWEVER: this is overcomplicated. The simpler reading matches:
        #   the hyper_connection_mixer is the OUTPUT norm, run AFTER the layer.
        #   pre_fc_norm_hidden scales the FLAT [hc_dim] view before an explicit
        #   mean-collapse: mean(h_norm.reshape(B,1,hc,D), axis=-2) = [B,1,D].
        #   THEN fc_hidden maps D->D.
        # This reading gives fc_hidden input dim = D = 2560, consistent with [2560,2560].
        B, T, hc, D = last_hidden_wide.shape
        h_norm = h_flat * self.pre_fc_norm_hidden.astype(h_flat.dtype)   # [B,1,hc_dim]
        h_collapsed = h_norm.reshape(B, T, hc, D).mean(axis=2)           # [B,1,D]
        h_proj = self.fc_hidden(h_collapsed)                              # [B,1,D]

        # Step 2: embedding branch — embed next token, plain RMSNorm, fc_embedding
        e_raw = embed_tokens(next_input_id)                              # [B,1,D]
        e_norm = mx.fast.rms_norm(e_raw, self.pre_fc_norm_embedding.astype(e_raw.dtype), self._eps)
        e_proj = self.fc_embedding(e_norm)                               # [B,1,D]

        # Step 3: fuse — ADD (see SPEC 2.1 for shape-based justification)
        fused = h_proj + e_proj                                          # [B,1,D]

        # Step 4: enter wide-residual form
        res = hc_init(fused, hc)                                         # [B,1,hc,D]

        # Step 5: run MTP decoder layer
        res = self.layers[0](res, cache=cache, pos_offset=pos_offset)   # [B,1,hc,D]

        # Step 6: output norm via MTP's own mixer
        out, _ = self.hyper_connection_mixer(res)                        # [B,1,D]

        # Step 7: logits via shared lm_head
        return lm_head(out)                                              # [B,1,vocab]


# --------------------------------------------------------------------------- cache factory
def make_mtp_cache() -> List[QSACache]:
    """One QSACache for the single MTP layer."""
    return [QSACache()]


# --------------------------------------------------------------------------- sanitize helpers
_MTP_PLUS1_SUFFIXES = (
    ".hc_norm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    ".indexer.q_layernorm",
    ".indexer.k_layernorm",
)

_MTP_BARE_GAMMA_SUFFIXES = (
    ".hc_norm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".indexer.q_layernorm.weight",
    ".indexer.k_layernorm.weight",
)


def sanitize_mtp_weights(
    raw_weights: Dict[str, mx.array],
    args: ModelArgs,
) -> Dict[str, mx.array]:
    """Transform raw ``mtp.*`` checkpoint keys into module-tree-compatible keys.

    Applies the same conventions as Model.sanitize, restricted to the mtp.* namespace:
      - hc_norm.weight -> hc_norm (bare gamma, +1 fold)
      - q_norm.weight / k_norm.weight / q_layernorm.weight / k_layernorm.weight
        -> bare gammas (drop .weight, +1 fold)
      - pre_fc_norm_hidden.weight -> pre_fc_norm_hidden (bare, NO +1 — see SPEC 2.5)
      - pre_fc_norm_embedding.weight -> pre_fc_norm_embedding (bare, NO +1)
      - indexer.index_qk_proj.weight [640,2560] -> index_q_proj / index_k_proj
      - experts.gate_up_proj -> switch_mlp.gate_proj / switch_mlp.up_proj
      - experts.down_proj -> switch_mlp.down_proj

    Args:
        raw_weights: dict where keys start with "mtp." (from the BF16 checkpoint or
                     the 4-bit MTP shard). These are NOT the full model weights.
        args: ModelArgs for the main model (shares dims with MTP).

    Returns:
        New dict ready for load_weights on an MTPDraft instance, with "mtp." prefix
        stripped (the caller does load_weights on the MTPDraft object whose root IS mtp).
    """
    n_q_idx = args.indexer_n_heads * args.indexer_head_dim  # 512
    n_experts = args.num_experts
    out: Dict[str, mx.array] = {}
    deferred_experts: Dict[str, Dict[str, mx.array]] = {}  # per-layer expert tensors

    for raw_k, v in raw_weights.items():
        if not raw_k.startswith("mtp."):
            raise ValueError(f"sanitize_mtp_weights: unexpected non-mtp key: {raw_k!r}")

        # Strip "mtp." prefix to get module-relative key
        k = raw_k[len("mtp."):]

        # indexer qk split
        if k.endswith(".self_attn.indexer.index_qk_proj.weight"):
            base = k[: -len(".index_qk_proj.weight")]
            if v.shape[0] < n_q_idx:
                raise ValueError(
                    f"{raw_k} has {v.shape[0]} rows, expected >= {n_q_idx}"
                )
            out[f"{base}.index_q_proj.weight"] = v[:n_q_idx]
            out[f"{base}.index_k_proj.weight"] = v[n_q_idx:]
            continue

        # bare gamma renames (drop .weight suffix)
        if any(k.endswith(sfx) for sfx in _MTP_BARE_GAMMA_SUFFIXES):
            new_k = k[: -len(".weight")]
            new_v = v + 1.0  # +1 fold for the covered norms
            out[new_k] = new_v
            continue

        # pre_fc_norm_* — bare gammas, WITH +1 fold.
        # Originally documented as NO +1 (SPEC 2.5), but empirical evidence shows
        # the BF16 checkpoint stores these as (gamma - 1): pre_fc_norm_embedding
        # mean=-0.76 would be nonsensical as a raw scale factor, but adding 1 gives
        # mean=0.24 which is a valid low-value gamma. The GGUF +1 rule covers all
        # *.norm.weight keys except linear_attn.norm.weight; although the suffix match
        # is "_hidden.weight" / "_embedding.weight" (not "norm.weight"), the stored
        # values confirm the (gamma-1) encoding. Apply +1 here to correct the load.
        if k in ("pre_fc_norm_hidden.weight", "pre_fc_norm_embedding.weight"):
            out[k[: -len(".weight")]] = v + 1.0
            continue

        # expert tensors — collect and process per-layer after the main loop
        if ".mlp.experts." in k:
            # key looks like "layers.0.mlp.experts.gate_up_proj" or ".down_proj"
            # extract the layer prefix up to ".mlp"
            mlp_idx = k.index(".mlp.experts.")
            layer_prefix = k[: mlp_idx + len(".mlp")]  # e.g. "layers.0.mlp"
            tensor_name = k[mlp_idx + len(".mlp.experts."):]  # "gate_up_proj" or "down_proj"
            # strip trailing .weight if present
            if tensor_name.endswith(".weight"):
                tensor_name = tensor_name[: -len(".weight")]
            if layer_prefix not in deferred_experts:
                deferred_experts[layer_prefix] = {}
            deferred_experts[layer_prefix][tensor_name] = v
            continue

        out[k] = v

    # process experts
    for layer_prefix, tensors in deferred_experts.items():
        if "gate_up_proj" in tensors:
            gate_up = tensors["gate_up_proj"]
            if gate_up.shape[0] != n_experts:
                raise ValueError(
                    f"mtp {layer_prefix} gate_up_proj expert axis {gate_up.shape[0]} != {n_experts}"
                )
            gate, up = split_gate_up(gate_up)
            out[f"{layer_prefix}.switch_mlp.gate_proj.weight"] = gate
            out[f"{layer_prefix}.switch_mlp.up_proj.weight"] = up
        if "down_proj" in tensors:
            down = tensors["down_proj"]
            if down.ndim != 3 or down.shape[0] != n_experts:
                raise ValueError(
                    f"mtp {layer_prefix} down_proj shape {down.shape} invalid"
                )
            out[f"{layer_prefix}.switch_mlp.down_proj.weight"] = down

    return out
