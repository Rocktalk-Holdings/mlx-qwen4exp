"""Full qwen4_exp (Qwen3.8-Flash-Next) model — the integrator that wires the four
finished sub-modules (hyper / ple / deltanet / attention / moe) into one MLX model.

Ground truth for the layer flow: ``llama.cpp/src/models/qwen4exp.cpp`` lines 295-345
(``llama_model_qwen4exp::graph::graph``). SPEC.md section 4.1 (layer flow), 5 (MLX
contract), 2 (tensor names), 3 (weight conventions).

Module tree (chosen so the checkpoint keys map with MINIMAL renaming — see ``sanitize``)::

    Model
      .model                                  (attr `model`, mirrors "model." key prefix)
        .language_model                       (mirrors "model.language_model.")
          .embed_tokens        nn.Embedding(vocab, 2560)
          .layers              [Qwen4ExpDecoderLayer x 48]
          .hyper_connection_mixer  HyperConnection(with_inject=False)   # the output norm
      .lm_head               nn.Linear(2560, vocab, bias=False)         (NOT tied)

Each decoder layer carries::

    .attn_hyper_connection   HyperConnection(with_inject=True)
    .mlp_hyper_connection    HyperConnection(with_inject=True)
    .linear_attn  (GatedDeltaNet)   XOR   .self_attn (QSAAttention)
    .mlp                     Qwen4ExpMoE
    .ple                     PLEBlock          (only on the PLE layer, 0-based idx 1)

The 51.2B-parameter n-gram table is NOT a module parameter. It is attached to the model
after load as ``self._ple_table`` (an mx.array or memmap-backed mx.array). The three I64
hash constants likewise live on the model (``self._ple_multipliers`` etc.), set by
``convert.py`` at load time. Without the table the PLE block is skipped (degraded mode)
with a single warning.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.cache import ArraysCache, KVCache

from .config import ModelArgs
from .hyper import HyperConnection, hc_combine, hc_init
from .ple import PLEBlock, ngram_rows
from .deltanet import GatedDeltaNet
from .attention import QSAAttention, IndexerCache, QSACache
from .moe import Qwen4ExpMoE, split_gate_up
from .mtp import MTPDraft, sanitize_mtp_weights, make_mtp_cache

__all__ = ["ModelArgs", "Model", "Qwen4ExpDecoderLayer", "Qwen4ExpModel", "QSACache"]


# --------------------------------------------------------------------------- decoder layer
class Qwen4ExpDecoderLayer(nn.Module):
    """One decoder layer. Attribute names are the exact checkpoint key segments
    (``attn_hyper_connection``, ``mlp_hyper_connection``, ``linear_attn`` / ``self_attn``,
    ``mlp``, ``ple``) so ``load_weights`` maps cleanly."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = args.is_linear(layer_idx)
        self.has_ple = layer_idx in args.ple_layers

        # both hyper-connections exist on every layer
        self.attn_hyper_connection = HyperConnection(args, with_inject=True)
        self.mlp_hyper_connection = HyperConnection(args, with_inject=True)

        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = QSAAttention(args)

        self.mlp = Qwen4ExpMoE(args)

        if self.has_ple:
            self.ple = PLEBlock(args)


# --------------------------------------------------------------------------- inner model
class Qwen4ExpModel(nn.Module):
    """Mirrors the ``model.language_model.*`` checkpoint namespace."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen4ExpDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = HyperConnection(args, with_inject=False)


class _LanguageModelWrapper(nn.Module):
    """Thin ``model`` container holding ``language_model`` so the tree matches
    ``model.language_model.layers.N.*`` exactly."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = Qwen4ExpModel(args)


# --------------------------------------------------------------------------- Model
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.model = _LanguageModelWrapper(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # -- MTP draft head (only allocated when args.load_mtp=True) -------------
        # Attribute exists unconditionally so state.setter works even when MTP is off.
        self.mtp: Optional[MTPDraft] = MTPDraft(args) if args.load_mtp else None
        # last wide residual tracked for MTP; updated every forward call
        self._last_hidden_wide: Optional[mx.array] = None

        # -- externally-attached PLE state (NOT module params) --------------------
        # the 51.2B-param n-gram table; None => degraded mode (PLE skipped, one warning)
        self._ple_table: Optional[mx.array] = None
        # the three I64 hash constants, stored as plain python/np after load
        self._ple_multipliers: Optional[np.ndarray] = None
        self._ple_offsets: Optional[np.ndarray] = None
        self._ple_vocab_sizes: Optional[np.ndarray] = None
        # per-generation running conv state for the PLE layer, keyed by nothing (single
        # stream): a dict {layer_idx: conv_state}. reset_ple_state() clears it.
        self._ple_conv: Dict[int, mx.array] = {}
        # per-generation last (ngram_size-1) token ids per sequence, [B, ngram_size-1]
        self._ple_prev: Optional[np.ndarray] = None
        self._ple_warned = False

    # ------------------------------------------------------------------ ple helpers
    @property
    def _ple_ready(self) -> bool:
        return (
            self._ple_table is not None
            and self._ple_multipliers is not None
            and self._ple_offsets is not None
            and self._ple_vocab_sizes is not None
        )

    def reset_ple_state(self) -> None:
        """Clear the per-generation PLE conv history and previous-token window. Call this
        before starting a fresh sequence so a new generation does not inherit stale
        n-gram context or conv state from the last one."""
        self._ple_conv = {}
        self._ple_prev = None

    def _ple_prev_for(self, inputs_np: np.ndarray) -> np.ndarray:
        """Return the [B, ngram_size-1] previous-token window for THIS call.

        On the first chunk of a sequence the predecessors do not exist -> all -1
        (``ngram_rows`` reads -1 as "missing" => EOS-cut). On subsequent chunks the
        window is whatever we stashed after the previous chunk. Threading this is exactly
        what makes a chunked prefill hash identically to a single-shot one (SPEC 5;
        ple.py landmine: prev_tokens is oldest-first, -1 = does-not-exist).
        """
        B = inputs_np.shape[0]
        n_prev = self.args.ngram_size - 1
        if self._ple_prev is None:
            return np.full((B, n_prev), -1, dtype=np.int64)
        if self._ple_prev.shape != (B, n_prev):
            raise ValueError(
                f"stale PLE prev-token window {self._ple_prev.shape} does not match "
                f"batch/ngram ({B}, {n_prev}); call reset_ple_state() between sequences"
            )
        return self._ple_prev

    def _ple_stash_prev(self, inputs_np: np.ndarray) -> None:
        """Update the previous-token window to the last (ngram_size-1) ids of this chunk.

        If the chunk is shorter than the window, carry the tail of the OLD window in front
        of it so a stream fed one token at a time still sees the right 2-back context.
        """
        n_prev = self.args.ngram_size - 1
        if n_prev <= 0:
            self._ple_prev = np.empty((inputs_np.shape[0], 0), dtype=np.int64)
            return
        old = self._ple_prev
        if old is None:
            old = np.full((inputs_np.shape[0], n_prev), -1, dtype=np.int64)
        combined = np.concatenate([old, inputs_np.astype(np.int64)], axis=1)
        self._ple_prev = combined[:, -n_prev:]

    # ------------------------------------------------------------------ forward
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs.ndim != 2:
            raise ValueError(f"inputs must be [B, T] int, got shape {inputs.shape}")
        B, T = inputs.shape
        lm = self.model.language_model
        hc = self.args.hc_count

        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = lm.embed_tokens(inputs)  # [B, T, D]

        if cache is None:
            cache = [None] * len(lm.layers)

        res = hc_init(h, hc)  # [B, T, hc, D]

        # host-side token ids for the PLE hash (only if a PLE layer is active + ready)
        run_ple = self._ple_ready
        if any(l.has_ple for l in lm.layers) and not run_ple and not self._ple_warned:
            warnings.warn(
                "PLE table/constants not attached (self._ple_table is None); running in "
                "DEGRADED mode with the PLE block skipped. Attach the table via convert.py "
                "for correct outputs.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._ple_warned = True

        inputs_np = None
        prev_np = None
        if run_ple:
            inputs_np = np.array(inputs)  # [B, T] host copy for the integer hash
            prev_np = self._ple_prev_for(inputs_np)

        for i, layer in enumerate(lm.layers):
            c = cache[i]

            # -- PLE runs BEFORE the attn hc_mix and rewrites the wide residual (cpp:304) --
            if layer.has_ple and run_ple:
                rows = ngram_rows(
                    inputs_np,
                    prev_np,
                    self._ple_multipliers,
                    self._ple_offsets,
                    self._ple_vocab_sizes,
                    self.args.ngram_size,
                    self.args.heads_per_ngram,
                    self.args.eos_token_id,
                )
                conv_state = self._ple_conv.get(i)
                res, new_conv = layer.ple(res, rows, self._ple_table, conv_state)
                self._ple_conv[i] = new_conv

            # -- attention block: hc_mix -> (delta|qsa) -> hc_combine --
            cur, inj = layer.attn_hyper_connection(res)
            if layer.is_linear:
                cur = layer.linear_attn(cur, mask=None, cache=c)
            else:
                # pos_offset comes from the KVCache offset BEFORE this step's update,
                # exactly as mlx_lm reads cache.offset for rope positioning.
                if c is not None:
                    pos_offset = c.offset
                    kv, idx = c.kv, c.idx
                else:
                    pos_offset = 0
                    kv, idx = None, None
                cur = layer.self_attn(
                    cur, cache=kv, idx_cache=idx, pos_offset=pos_offset
                )
            res = hc_combine(res, cur, inj, hc)

            # -- mlp block: hc_mix -> moe -> hc_combine --
            cur, inj = layer.mlp_hyper_connection(res)
            cur = layer.mlp(cur)
            res = hc_combine(res, cur, inj, hc)

        # -- stash the wide residual for MTP BEFORE the terminal mixer collapses it --
        # The MTP draft head needs grouped_rms_norm(res) which requires the [B,T,hc,D]
        # view. We only keep the LAST position's slice (decode mode, T=1 typically).
        if self.mtp is not None:
            self._last_hidden_wide = res

        # -- the terminal mixer is the output norm (no inject) --
        out = lm.hyper_connection_mixer(res)[0]  # [B, T, D]

        # stash the tail of this chunk so the next chunk hashes with the right context
        if run_ple:
            self._ple_stash_prev(inputs_np)

        return self.lm_head(out)

    # ------------------------------------------------------------------ contract
    @property
    def layers(self):
        return self.model.language_model.layers

    def make_cache(self):
        """One cache object per layer, typed by layer position:
          * linear (GatedDeltaNet) layer -> ArraysCache(size=2)   ([0]=conv, [1]=recurrent)
          * QSA (full-attention) layer   -> QSACache               (.kv KVCache, .idx IndexerCache)
        """
        out = []
        for layer in self.model.language_model.layers:
            if layer.is_linear:
                out.append(ArraysCache(size=2))
            else:
                out.append(QSACache())
        return out

    # ------------------------------------------------------------------ sanitize
    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map raw HF checkpoint keys onto this module tree.

        The tree deliberately mirrors ``model.language_model.*`` so NO prefix stripping is
        needed for the language weights (a) — the outer ``model`` attr absorbs ``model.``
        and ``language_model`` absorbs ``language_model.``. lm_head stays at the root.

        Duties (SPEC 3 + module reports); each documented with its evidence:

          a) drop ``model.visual.*`` and ``mtp.*`` junk (SPEC 2 "Ignore for now").
          b) drop the PLE I64 constants and ngram-shard tensors — handled by convert.py,
             not loaded as module params (keys containing ``.ple.ple_embedding.``).
          c) NO renaming of the language prefix (tree matches — see above).
          d) ``...hc_norm.weight`` -> ``...hc_norm`` (bare mx.array, not an nn.Module).
          e) +1 folding for zero-centred gammas. DEFINITIVE RULE (derived from the GGUF
             converter, which we must replicate since we load raw HF weights):
               - BASE rule (conversion/qwen.py:394): any key ending ``norm.weight`` gets
                 +1, EXCEPT ``linear_attn.norm.weight``. This is inherited by
                 Qwen4ExpTextModel via _LinearAttentionVReorderBase -> Qwen3NextModel ->
                 Qwen2MoeModel. It covers:  hc_norm.weight (all 3 sites),
                 self_attn.q_norm.weight, self_attn.k_norm.weight.  (cpp:209 confirms the
                 hc gammas were folded to (1+w).)
               - EXCLUSION: linear_attn.norm.weight -> NO +1 (qwen.py:394 explicit
                 ``and not name.endswith("linear_attn.norm.weight")``).
               - OVERRIDE (conversion/qwen4exp.py:140-142): these do NOT end in
                 ``norm.weight`` so the base rule misses them; the qwen4exp converter adds
                 +1 explicitly:  ple.norm_key.weight, ple.norm_query.weight,
                 ple.norm_conv.weight, indexer.q_layernorm.weight, indexer.k_layernorm.weight.
          f) split fused expert ``mlp.experts.gate_up_proj`` -> switch_mlp.gate_proj/up_proj;
             ``mlp.experts.down_proj`` -> switch_mlp.down_proj (already [E,out,in]).
          g) split ``self_attn.indexer.index_qk_proj.weight`` [640,2560] -> index_q_proj
             (rows 0:512) / index_k_proj (rows 512:640).
          h) conv1d: linear_attn -> [C,1,K] moveaxis(2,1) (qwen3_next idiom the deltanet
             module reuses); ple -> [C,1,K] squeeze to [C,K] (ple.py wants [C,K]).
          i) q_norm/k_norm/linear_attn.norm are bare mx.array gammas in their modules; the
             checkpoint stores them as ``<name>.weight``. The module attribute for q_norm/
             k_norm is a bare array named ``q_norm``/``k_norm`` (QSAAttention), so the
             ``.weight`` suffix is dropped. linear_attn.norm is a Qwen4ExpRMSNormGated
             module whose param IS ``norm.weight``, so it keeps the suffix.
        """
        # nothing to do for an already-sanitized dict (idempotent re-entry)
        if not weights:
            return weights

        eps_add = 1.0
        n_layers = self.args.num_hidden_layers
        n_experts = self.args.num_experts
        n_q_idx = self.args.indexer_n_heads * self.args.indexer_head_dim  # 512

        out: Dict[str, mx.array] = {}

        # ---- +1 folding rule, evaluated on the ALREADY-RENAMED key ---------------
        # After the bare-gamma rename above, hc_norm/q_norm/k_norm lose ".weight" and the
        # ple/indexer gammas lose ".weight" too, so we match the renamed suffixes here.
        # The three sets:
        #   BASE (qwen.py:394): any "*norm.weight" except "linear_attn.norm.weight".
        #     Renamed -> hc_norm / self_attn.q_norm / self_attn.k_norm.
        #   OVERRIDE (qwen4exp.py:140-142): ple.norm_key/query/conv, indexer.q/k_layernorm.
        #     Renamed -> norm_key / norm_query / norm_conv / q_layernorm / k_layernorm.
        PLUS1_RENAMED = (
            ".hc_norm",
            ".self_attn.q_norm",
            ".self_attn.k_norm",
            ".ple.norm_key",
            ".ple.norm_query",
            ".ple.norm_conv",
            ".indexer.q_layernorm",
            ".indexer.k_layernorm",
        )

        def fold_plus_one(k: str, v: mx.array) -> bool:
            if v.ndim != 1:
                return False
            if k.endswith("linear_attn.norm.weight"):
                return False  # EXCLUSION (qwen.py:394) — never fold the GDN output norm
            return any(k.endswith(sfx) for sfx in PLUS1_RENAMED)

        # Collect mtp.* tensors separately; route them via sanitize_mtp_weights
        # if load_mtp is True, otherwise drop them (legacy behaviour).
        mtp_raw: Dict[str, mx.array] = {}
        for k, v in weights.items():
            # (a) drop visual junk; conditionally drop mtp.*
            if k.startswith("model.visual.") or k.startswith("model.mtp."):
                continue
            if k.startswith("mtp."):
                if self.args.load_mtp:
                    mtp_raw[k] = v
                continue
            # (b) drop PLE I64 constants + ngram shards (handled by convert.py)
            if ".ple.ple_embedding." in k:
                continue
            out[k] = v

        weights = out
        out = {}

        # ---- per-key transforms -------------------------------------------------
        for k, v in weights.items():
            new_k = k
            new_v = v

            # (g) split the indexer qk projection
            if k.endswith(".self_attn.indexer.index_qk_proj.weight"):
                base = k[: -len(".index_qk_proj.weight")]
                if v.shape[0] < n_q_idx:
                    raise ValueError(
                        f"{k} has {v.shape[0]} rows, expected >= {n_q_idx} for the q split"
                    )
                out[f"{base}.index_q_proj.weight"] = v[:n_q_idx]
                out[f"{base}.index_k_proj.weight"] = v[n_q_idx:]
                continue

            # (d/i) bare-mx.array gammas lose their checkpoint ".weight" suffix:
            #   hc_norm (HyperConnection), self_attn.q_norm/k_norm (QSAAttention),
            #   ple.norm_key/query/conv (PLEBlock),
            #   indexer.q_layernorm/k_layernorm (QSAIndexer).
            # linear_attn.norm and switch_mlp/shared_expert are real nn.Modules -> keep ".weight".
            _BARE_GAMMA_SUFFIXES = (
                ".hc_norm.weight",
                ".self_attn.q_norm.weight",
                ".self_attn.k_norm.weight",
                ".ple.norm_key.weight",
                ".ple.norm_query.weight",
                ".ple.norm_conv.weight",
                ".indexer.q_layernorm.weight",
                ".indexer.k_layernorm.weight",
            )
            if any(k.endswith(sfx) for sfx in _BARE_GAMMA_SUFFIXES):
                new_k = k[: -len(".weight")]

            # (h) conv1d transforms
            if k.endswith(".linear_attn.conv1d.weight"):
                # qwen3_next idiom: [C,1,K] -> moveaxis(2,1) -> [C,K,1] shaped as nn.Conv1d wants
                if v.ndim == 3 and v.shape[-1] != 1:
                    new_v = v.moveaxis(2, 1)
            elif k.endswith(".ple.conv1d.weight"):
                # ple.py's conv1d is a BARE mx.array named `conv1d` (not an nn.Conv1d), and
                # it wants shape [C, K]: squeeze the singleton middle axis of [C,1,K] and
                # drop the checkpoint's ".weight" suffix so it maps to the bare attribute.
                if v.ndim == 3:
                    new_v = v.reshape(v.shape[0], v.shape[2])
                new_k = k[: -len(".weight")]

            # (e) +1 folding
            if fold_plus_one(new_k, new_v):
                new_v = new_v + eps_add

            out[new_k] = new_v

        weights = out
        out = dict(weights)

        # (f) split/collect the fused experts, per layer, into stacked SwitchLinear weights
        for l in range(n_layers):
            prefix = f"model.language_model.layers.{l}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            down_key = f"{prefix}.experts.down_proj"
            # allow the optional trailing ".weight" the HF export sometimes carries
            if gate_up_key not in out and f"{gate_up_key}.weight" in out:
                gate_up_key = f"{gate_up_key}.weight"
            if down_key not in out and f"{down_key}.weight" in out:
                down_key = f"{down_key}.weight"

            if gate_up_key in out:
                gate_up = out.pop(gate_up_key)
                gate, up = split_gate_up(gate_up)
                # SwitchLinear weight is [E, out, in]; gate/up map 2560->640 => [E,640,2560]
                if gate.shape[0] != n_experts:
                    raise ValueError(
                        f"{gate_up_key} expert axis {gate.shape[0]} != {n_experts}"
                    )
                out[f"{prefix}.switch_mlp.gate_proj.weight"] = gate
                out[f"{prefix}.switch_mlp.up_proj.weight"] = up

            if down_key in out:
                down = out.pop(down_key)
                # down maps 640->2560 => [E,2560,640] already [E,out,in]; assert it
                if down.ndim != 3 or down.shape[0] != n_experts:
                    raise ValueError(
                        f"{down_key} shape {down.shape} not [{n_experts}, out, in]"
                    )
                exp_out = self.args.hidden_size
                exp_in = self.args.moe_intermediate_size
                if down.shape[1:] != (exp_out, exp_in):
                    raise ValueError(
                        f"{down_key} shape {tuple(down.shape)} != "
                        f"[{n_experts}, {exp_out}, {exp_in}] (E, hidden, moe_inter)"
                    )
                out[f"{prefix}.switch_mlp.down_proj.weight"] = down

        # (mtp) merge sanitised mtp.* weights into output using "mtp." prefix so
        # load_weights can find the self.mtp module. Keys come out of
        # sanitize_mtp_weights WITHOUT the "mtp." prefix; re-add it here.
        if self.args.load_mtp and mtp_raw:
            mtp_sanitized = sanitize_mtp_weights(mtp_raw, self.args)
            for mtp_k, mtp_v in mtp_sanitized.items():
                out[f"mtp.{mtp_k}"] = mtp_v

        return out

    # ------------------------------------------------------------------ quantization
    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Keep the token embedding and the (untied) output head at 8-bit, as the
            # reference GGUF does (token_embd/output = Q8_0). 4-bit on these destroys the
            # small embedding entries and the logit projection, corrupting the whole stack.
            if path.endswith("embed_tokens") or path == "lm_head" or path.endswith(".lm_head"):
                return {"group_size": 64, "bits": 8}
            # never quantize the norms / gates / gammas: they are tiny and precision-critical
            if (
                path.endswith("hc_norm")
                or path.endswith("q_norm")
                or path.endswith("k_norm")
                or path.endswith("norm_key")
                or path.endswith("norm_query")
                or path.endswith("norm_conv")
                or path.endswith("q_layernorm")
                or path.endswith("k_layernorm")
                or path.endswith("norm.weight")
                or path.endswith("A_log")
                or path.endswith("dt_bias")
            ):
                return False
            # route the routing gates to 8-bit, as qwen3_next does
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
