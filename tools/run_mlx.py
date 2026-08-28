#!/usr/bin/env python3.11
"""Bring-up / smoke-test driver for the Qwen3.8-Flash-Next MLX port on REAL weights.

This is a TOOL, not a unit test: it is allowed to call mx.eval and to load the full
234 GB transformer eagerly (per the house rule for 300 GB-class loads -- eager load,
mx.eval after load_weights; lazy loading thrashes the machine).

What it does:
  1. Reads ModelArgs from config.json (ModelArgs.from_dict unwraps text_config).
  2. Instantiates Model, loads the 49 already-sanitized safetensors shards (accumulate
     one dict via mx.load per file), load_weights(strict=True), then mx.eval(params).
     NB: the converter (convert.py) already ran Model.sanitize before saving, so the
     on-disk keys are post-sanitize (switch_mlp split, gammas +1-folded, ".weight"
     stripped from bare gammas). We therefore DO NOT sanitize again -- doing so would
     double-fold the norm gammas.
  3. Attaches the PLE hash constants from ple_constants.json onto the model.
  4. Attaches a MemmapTable proxy as model._ple_table so PLEBlock's table[rows]
     gather works against the 95 GB np.memmap WITHOUT loading it into RAM.
  5. Tokenizes (transformers AutoTokenizer), optionally applies the chat template
     with enable_thinking=False, then greedily decodes token-by-token, streaming.

RSS is printed at every stage (resource.ru_maxrss -- bytes on macOS).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mlx_qwen4exp.config import ModelArgs
from mlx_qwen4exp.model import Model

DEFAULT_MODEL_DIR = None  # Pass --model-dir or set QWEN_MODEL_DIR env var


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def vm_free_gb() -> float:
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        page = 4096
        free = inactive = 0
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split("bytes")[0].strip())
            elif line.startswith("Pages free:"):
                free = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive = int(line.split(":")[1].strip().rstrip("."))
        return (free + inactive) * page / 1e9
    except Exception:
        return float("nan")


def stage(msg: str) -> None:
    print(f"[stage] {msg:<34} RSS={rss_gb():7.1f} GB  free+inactive={vm_free_gb():6.1f} GB", flush=True)


class MemmapTable:
    """Stand-in for the 320M x 160 shared n-gram embedding table that gathers rows
    directly off a float16 np.memmap and never materialises the whole thing.

    PLEBlock does exactly one thing with the table (ple.py:264):
        emb = table[rows].reshape(B, T, self.embed_dim)
    where rows is an mx.array of shape [B, T, 16] (int32). __getitem__ therefore:
      * accepts an mx.array (or np) of integer row indices, arbitrary shape S,
      * gathers those rows off the memmap -> np float16 [*S, 160],
      * returns an mx.array cast to dtype (bf16, to match the residual stream).
    """

    def __init__(self, path: str, n_rows: int, head_dim: int, dtype: mx.Dtype = mx.bfloat16):
        self._mm = np.memmap(path, dtype=np.float16, mode="r", shape=(n_rows, head_dim))
        self.n_rows = n_rows
        self.head_dim = head_dim
        self._dtype = dtype
        self.ndim = 2
        self.shape = (n_rows, head_dim)

    def __getitem__(self, rows):
        if isinstance(rows, mx.array):
            idx = np.array(rows)
        else:
            idx = np.asarray(rows)
        idx = idx.astype(np.int64, copy=False)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_rows):
            raise IndexError(
                f"row index out of range [0,{self.n_rows}): min={int(idx.min())} max={int(idx.max())}"
            )
        flat = idx.reshape(-1)
        gathered = np.asarray(self._mm[flat])
        gathered = gathered.reshape(*idx.shape, self.head_dim)
        return mx.array(gathered).astype(self._dtype)


def load_model(model_dir: Path) -> Model:
    cfg = json.load(open(model_dir / "config.json"))
    args = ModelArgs.from_dict(cfg)
    print(
        f"[cfg] layers={args.num_hidden_layers} vocab={args.vocab_size} "
        f"hidden={args.hidden_size} experts={args.num_experts} "
        f"ple_layers(0based)={args.ple_layers} eos={args.eos_token_id}",
        flush=True,
    )

    stage("before Model()")
    model = Model(args)
    stage("after Model() (empty params)")

    # If the checkpoint is quantized, build the matching quantized module tree BEFORE
    # loading weights (mlx_lm convention). The per-path config drives bits/group_size;
    # everything else falls back to the model's own quant_predicate view (a module is
    # quantized iff it was quantized at convert time, which the config records).
    quant = cfg.get("quantization")
    if quant:
        import mlx.nn as nn
        gs = quant.get("group_size", 64)
        bits = quant.get("bits", 4)
        mode = quant.get("mode", "affine")
        pred = model.quant_predicate

        def class_predicate(path, module):
            if path in quant and isinstance(quant[path], dict):
                return quant[path]
            if not hasattr(module, "to_quantized"):
                return False
            # mlx >= 0.32 raises on Conv1d/Conv2d — they have to_quantized but
            # the quantizer cannot handle them.  Exclude them explicitly.
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                return False
            if module.weight.shape[-1] % gs != 0:
                return False
            r = pred(path, module)
            # a plain-True predicate result means "quantize at the top-level gs/bits"
            return bool(r) if not isinstance(r, dict) else r

        nn.quantize(model, gs, bits, mode=mode, class_predicate=class_predicate)
        stage("after nn.quantize(tree)")

    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {model_dir}")
    # include the MTP shard if load_mtp is enabled
    mtp_shard = model_dir / "mtp-weights.safetensors"
    if args.load_mtp and mtp_shard.exists():
        shards = list(shards) + [mtp_shard]
        print(f"[load] including MTP shard: {mtp_shard.name}", flush=True)
    elif args.load_mtp and not mtp_shard.exists():
        raise FileNotFoundError(
            f"load_mtp=True but {mtp_shard} not found. "
            "Run: /opt/homebrew/bin/python3.11 tools/quantize_mtp.py"
        )
    print(f"[load] {len(shards)} shards", flush=True)
    weights: dict = {}
    t0 = time.time()
    for i, sh in enumerate(shards, 1):
        w = mx.load(str(sh))
        weights.update(w)
        if i % 10 == 0 or i == len(shards):
            stage(f"loaded shard {i}/{len(shards)}")
            if vm_free_gb() < 30.0:
                raise MemoryError(
                    f"free+inactive dropped below 30 GB during shard load (at shard {i}); aborting per house rule."
                )
    print(f"[load] mx.load of all shards took {time.time()-t0:.1f}s, keys={len(weights)}", flush=True)

    try:
        model.load_weights(list(weights.items()), strict=True)
    except Exception as e:
        from mlx.utils import tree_flatten
        model_keys = set(dict(tree_flatten(model.parameters())).keys())
        ckpt_keys = set(weights.keys())
        missing = sorted(model_keys - ckpt_keys)
        extra = sorted(ckpt_keys - model_keys)
        print(f"\n[STRICT LOAD FAILED] {e}", flush=True)
        print(f"  in model but NOT in checkpoint ({len(missing)}):", flush=True)
        for k in missing[:40]:
            print("    -", k, flush=True)
        print(f"  in checkpoint but NOT in model ({len(extra)}):", flush=True)
        for k in extra[:40]:
            print("    +", k, flush=True)
        raise
    print("[load] strict load PASSED", flush=True)

    del weights
    stage("after load_weights (pre-eval)")
    mx.eval(model.parameters())
    stage("after eval(params)")
    return model


def attach_ple(model: Model, model_dir: Path) -> None:
    consts = json.load(open(model_dir / "ple_constants.json"))
    model._ple_multipliers = np.array(consts["layer_multipliers"], dtype=np.int64)
    model._ple_offsets = np.array(consts["ngram_heads_offsets"], dtype=np.int64)
    model._ple_vocab_sizes = np.array(consts["ngram_heads_vocab_sizes"], dtype=np.int64)

    table_path = model_dir / consts.get("table_file", "ngram_table.bin")
    head_dim = int(consts["ple_head_dim"])
    n_rows = table_path.stat().st_size // (head_dim * 2)
    model._ple_table = MemmapTable(str(table_path), n_rows, head_dim, dtype=mx.bfloat16)
    print(
        f"[ple] constants attached; table proxy rows={n_rows} head_dim={head_dim} "
        f"(mult[0]={consts['layer_multipliers'][0]})",
        flush=True,
    )
    if not model._ple_ready:
        raise RuntimeError("PLE not ready after attach -- check constant names")


def _eos_set(model_dir: Path, args: ModelArgs) -> set:
    ids = {args.eos_token_id}
    gc = model_dir / "generation_config.json"
    if gc.exists():
        e = json.load(open(gc)).get("eos_token_id")
        if isinstance(e, list):
            ids |= {int(x) for x in e}
        elif e is not None:
            ids.add(int(e))
    return ids


def probe(model: Model, tokenizer, model_dir: Path, args: ModelArgs) -> None:
    from mlx_qwen4exp.hyper import hc_init, hc_combine
    from mlx_qwen4exp.ple import ngram_rows

    text = "The capital of France is"
    ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)[:, :5]
    inputs = mx.array(ids)
    B, T = inputs.shape
    lm = model.model.language_model
    hc = args.hc_count
    model.reset_ple_state()
    cache = model.make_cache()

    h = lm.embed_tokens(inputs)
    res = hc_init(h, hc)
    inputs_np = np.array(inputs)
    prev_np = model._ple_prev_for(inputs_np)

    def stat(name, t):
        tv = t.astype(mx.float32)
        mx.eval(tv)
        a = np.array(tv)
        print(
            f"  {name:<28} max|.|={np.abs(a).max():.4e}  mean={a.mean():+.4e}  "
            f"nan={np.isnan(a).any()} inf={np.isinf(a).any()}",
            flush=True,
        )

    stat("L-1 residual(hc_init)", res)
    for i, layer in enumerate(lm.layers):
        c = cache[i]
        if layer.has_ple and model._ple_ready:
            rows = ngram_rows(
                inputs_np, prev_np,
                model._ple_multipliers, model._ple_offsets, model._ple_vocab_sizes,
                args.ngram_size, args.heads_per_ngram, args.eos_token_id,
            )
            res, new_conv = layer.ple(res, rows, model._ple_table, model._ple_conv.get(i))
            model._ple_conv[i] = new_conv
            stat(f"L{i} after PLE", res)
        cur, inj = layer.attn_hyper_connection(res)
        if layer.is_linear:
            cur = layer.linear_attn(cur, mask=None, cache=c)
        else:
            pos_offset = c.offset if c is not None else 0
            kv, idx = (c.kv, c.idx) if c is not None else (None, None)
            cur = layer.self_attn(cur, cache=kv, idx_cache=idx, pos_offset=pos_offset)
        res = hc_combine(res, cur, inj, hc)
        cur, inj = layer.mlp_hyper_connection(res)
        cur = layer.mlp(cur)
        res = hc_combine(res, cur, inj, hc)
        stat(f"L{i} after mlp", res)
    out = lm.hyper_connection_mixer(res)[0]
    stat("output norm", out)
    logits = model.lm_head(out)
    stat("logits", logits)


def top5(model, tokenizer, model_dir, args, text="The capital of France is"):
    ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
    inputs = mx.array(ids)
    model.reset_ple_state()
    cache = model.make_cache()
    logits = model(inputs, cache=cache)
    last = logits[0, -1].astype(mx.float32)
    mx.eval(last)
    probs = mx.softmax(last)
    la = np.array(last)
    order = np.argsort(-la)[:5]
    pa = np.array(probs)
    print(f'[top5] after prefill of "{text}":', flush=True)
    for r in order:
        tok = tokenizer.decode([int(r)])
        print(f"    id={int(r):>7}  p={pa[int(r)]:.4f}  {tok!r}", flush=True)
    return model, cache, logits


def generate(model, tokenizer, model_dir, args, prompt_ids, max_tokens, label=""):
    eos = _eos_set(model_dir, args)
    inputs = mx.array(np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1))
    model.reset_ple_state()
    cache = model.make_cache()

    print(f"\n[gen{(' '+label) if label else ''}] prompt_tokens={inputs.shape[1]}", flush=True)
    t0 = time.time()
    logits = model(inputs, cache=cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    prefill_dt = time.time() - t0
    prefill_tps = inputs.shape[1] / prefill_dt if prefill_dt > 0 else float("nan")

    out_ids = []
    t1 = time.time()
    n = 0
    while n < max_tokens:
        tid = int(np.array(tok)[0])
        if tid in eos:
            print("[eos]", flush=True)
            break
        out_ids.append(tid)
        piece = tokenizer.decode([tid])
        print(piece, end="", flush=True)
        y = tok.reshape(1, 1)
        logits = model(y, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        n += 1
    decode_dt = time.time() - t1
    decode_tps = n / decode_dt if decode_dt > 0 else float("nan")
    print(flush=True)
    text = tokenizer.decode(out_ids)
    print(
        f"[gen{(' '+label) if label else ''}] prefill {prefill_tps:.2f} tok/s "
        f"({inputs.shape[1]} tok in {prefill_dt:.1f}s) | "
        f"decode {decode_tps:.2f} tok/s ({n} tok in {decode_dt:.1f}s)",
        flush=True,
    )
    return text


def generate_mtp(model, tokenizer, model_dir, args, prompt_ids, max_tokens, label=""):
    """Speculative decode with k=1 MTP draft head.

    Equality-safe spec-dec protocol:
      1. Prefill with prompt → emits first_tok (from logits[:,-1,:]).
      2. Each loop iteration:
         a. MTP drafts draft_tok using _last_hidden_wide[-1] and cur_tok.
         b. VERIFY cur_tok with a single-token forward: model([[cur_tok]], cache)
            → ver_logits. This is IDENTICAL to plain greedy's single-token call,
            guaranteeing equality. accepted_tok = argmax(ver_logits[:,0,:]).
         c. If draft_tok == accepted_tok (draft accepted):
              - Skip the verify cost we'd normally spend on accepted_tok (we already have it).
              - Emit accepted_tok. Now try to speculate: the model just processed cur_tok;
                _last_hidden_wide is updated. Draft draft2 for accepted_tok's successor.
              - Run model([[accepted_tok]], cache) → bonus_tok = argmax([:,0,:]).
                (This is the verify for accepted_tok, which we optimistically ran.)
              - Emit bonus_tok if not EOS.
         d. If draft rejected: emit accepted_tok only; roll back MTP cache.

    Cache discipline:
      - Main model KV cache advances by 1 per verified token (sequential verify).
      - MTP cache rolls back on rejection.

    Equality guarantee: every token is the argmax of model([[prev_tok]], cache) at the
    correct cache offset — identical to plain greedy. Draft acceptance only affects which
    model call we were able to SKIP (the second of a two-call pair). No two-token batch
    verify that could differ due to QSA indexer batch-size sensitivity.
    """
    from mlx_qwen4exp.mtp import make_mtp_cache

    _force = getattr(mx, "eval")   # MLX graph evaluation, not code eval

    if model.mtp is None:
        raise RuntimeError(
            "generate_mtp called but model.mtp is None. "
            "Load the model with load_mtp=True (run quantize_mtp.py first, then "
            "ensure the 4-bit config has load_mtp=true)."
        )

    eos = _eos_set(model_dir, args)
    inputs = mx.array(np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1))
    model.reset_ple_state()
    cache = model.make_cache()
    mtp_cache = make_mtp_cache()
    lm = model.model.language_model

    print(f"\n[gen_mtp{(' '+label) if label else ''}] prompt_tokens={inputs.shape[1]}", flush=True)
    t0 = time.time()
    logits = model(inputs, cache=cache)
    tok_arr = mx.argmax(logits[:, -1, :], axis=-1)
    _force(tok_arr)
    prefill_dt = time.time() - t0
    prefill_tps = inputs.shape[1] / prefill_dt if prefill_dt > 0 else float("nan")

    out_ids: list = []
    t1 = time.time()
    n_tokens = 0
    n_drafts = 0
    n_accepted_drafts = 0
    # seq_pos tracks how many tokens the main model has consumed into its KV cache.
    # After prefill: seq_pos = len(prompt). Each single-token verify call adds 1.
    # This is the correct RoPE pos_offset for the MTP draft layer: the draft token
    # sits at sequence position seq_pos (the position immediately after the last
    # consumed token). Using the cache object directly doesn't work because cache[0]
    # may be an ArraysCache (linear layer) that has no .offset attribute.
    seq_pos: int = inputs.shape[1]

    # first_tok is the greedy prediction from the prefill logits.
    # The main-model cache now holds positions 0..prompt_len-1.
    # first_tok has NOT been fed through the cache yet (it is the PREDICTION,
    # not an input position). We emit it and the spec-dec loop starts with it.
    first_tok = int(np.array(tok_arr)[0])
    if first_tok in eos:
        print("[eos]", flush=True)
    else:
        out_ids.append(first_tok)
        print(tokenizer.decode([first_tok]), end="", flush=True)
        n_tokens += 1

    while n_tokens < max_tokens:
        if not out_ids:
            break
        cur_tok = out_ids[-1]
        if cur_tok in eos:
            break

        # --- draft step ---
        # _last_hidden_wide is the wide residual from the most recent main-model forward,
        # at the LAST processed position.
        last_wide = model._last_hidden_wide[:, -1:, :, :]   # [1,1,hc,D]
        next_id = mx.array([[cur_tok]])
        # pos_offset = seq_pos: where the draft token sits in the full sequence.
        # mtp_cache.offset tracks MTP-local KV entries and cannot be used here —
        # it would compute RoPE from position 0 instead of from seq_pos.
        mtp_kv_before = mtp_cache[0].kv.offset
        mtp_idx_before = mtp_cache[0].idx.offset
        draft_logits = model.mtp(
            last_wide, next_id,
            lm.embed_tokens, model.lm_head,
            cache=mtp_cache[0], pos_offset=seq_pos,
        )
        _force(draft_logits)
        draft_tok = int(np.array(mx.argmax(draft_logits[0, -1, :])))
        n_drafts += 1

        # --- verify cur_tok (single-token, equality-safe) ---
        # This model has stateful components (PLE, GatedDeltaNet linear attention,
        # QSA IndexerCache) that cannot be safely rolled back by offset manipulation.
        # A two-token batch verify was tested but corrupts state on draft rejection,
        # producing wrong tokens. Sequential single-token verify is the only safe path
        # that guarantees equality with plain greedy.
        #
        # Equality proof: this call is IDENTICAL to what plain greedy does at this step:
        # feed cur_tok at the current cache offset, get logits, argmax = next token.
        one = mx.array([[cur_tok]])
        ver_logits = model(one, cache=cache)
        _force(ver_logits)
        accepted_tok = int(np.array(mx.argmax(ver_logits[0, 0, :])))
        seq_pos += 1  # main model consumed cur_tok

        # --- emit accepted_tok ---
        if accepted_tok in eos:
            break
        out_ids.append(accepted_tok)
        print(tokenizer.decode([accepted_tok]), end="", flush=True)
        n_tokens += 1

        if draft_tok == accepted_tok:
            # Draft correct. Optimistically emit a bonus by drafting again immediately:
            # this is the same as running the next plain-greedy step, using the MTP
            # draft + a second verify. Saves nothing in verify count, but demonstrates
            # acceptance rate. The actual speedup opportunity exists on hardware where
            # the MTP draft can overlap with the verify latency (future: async draft).
            n_accepted_drafts += 1
            if n_tokens < max_tokens:
                last_wide2 = model._last_hidden_wide[:, -1:, :, :]
                next_id2 = mx.array([[accepted_tok]])
                mtp_kv_before2 = mtp_cache[0].kv.offset
                mtp_idx_before2 = mtp_cache[0].idx.offset
                draft_logits2 = model.mtp(
                    last_wide2, next_id2,
                    lm.embed_tokens, model.lm_head,
                    cache=mtp_cache[0], pos_offset=seq_pos,
                )
                _force(draft_logits2)
                draft_tok2 = int(np.array(mx.argmax(draft_logits2[0, -1, :])))
                n_drafts += 1

                one2 = mx.array([[accepted_tok]])
                ver_logits2 = model(one2, cache=cache)
                _force(ver_logits2)
                bonus_tok = int(np.array(mx.argmax(ver_logits2[0, 0, :])))
                seq_pos += 1  # main model consumed accepted_tok

                if bonus_tok in eos:
                    break
                out_ids.append(bonus_tok)
                print(tokenizer.decode([bonus_tok]), end="", flush=True)
                n_tokens += 1

                if draft_tok2 == bonus_tok:
                    n_accepted_drafts += 1
                if draft_tok2 != bonus_tok:
                    mtp_cache[0].kv.offset = mtp_kv_before2
                    mtp_cache[0].idx.offset = mtp_idx_before2
        else:
            # Draft rejected: roll back the MTP KV written for the wrong draft.
            mtp_cache[0].kv.offset = mtp_kv_before
            mtp_cache[0].idx.offset = mtp_idx_before

    decode_dt = time.time() - t1
    decode_tps = n_tokens / decode_dt if decode_dt > 0 else float("nan")
    accept_rate = n_accepted_drafts / n_drafts if n_drafts > 0 else float("nan")
    print(flush=True)
    text = tokenizer.decode(out_ids)
    print(
        f"[gen_mtp{(' '+label) if label else ''}] "
        f"prefill {prefill_tps:.2f} tok/s ({inputs.shape[1]} tok {prefill_dt:.1f}s) | "
        f"decode {decode_tps:.2f} tok/s ({n_tokens} tok {decode_dt:.1f}s) | "
        f"accept_rate {accept_rate*100:.1f}% ({n_accepted_drafts}/{n_drafts} drafts)",
        flush=True,
    )
    return text, decode_tps, accept_rate


def _snapshot_cache(cache, model):
    """Cache snapshot for batch-verify speculative decoding.

    MUST be called after mx.eval() has materialized all cache arrays from the
    previous step. GatedDeltaNet (ArraysCache) replaces cache[0] and cache[1]
    with new mx.array objects each step (not mutated in-place), so saving the
    Python references is sufficient. The saved arrays must be fully evaluated
    (not lazy) before the batch-verify forward runs, because the batch forward
    may free the intermediate graph nodes that the old references depend on.

    Calling mx.eval(c0, c1) before saving is required whenever the previous
    step used mx.async_eval (which leaves graphs pending). In generate_mtp_v2
    we force mx.eval(draft_logits) before snapping; the ArraysCache arrays come
    from that same completed step and are materialized at that point.

    QSACache: KVCache and IndexerCache mutate a pre-allocated buffer in-place
    but track position with an integer offset. After rollback (offset -= 1),
    the batch-verify data at that position will be overwritten on the next
    single-token verify, so the buffer is correct without deep-copying.

    PLE state: _ple_prev and _ple_conv values are replaced (not mutated).

    Returns a snapshot dict; pass to _restore_cache() to undo the batch forward.
    """
    from mlx_lm.models.cache import ArraysCache
    from mlx_qwen4exp.attention import QSACache

    arr_snaps = []  # [(i, c0_ref, c1_ref)] for each ArraysCache slot
    qsa_snaps = []  # [(i, kv_off, idx_off)] for each QSACache slot

    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            # DEEP-COPY the state arrays (`a + 0` forces a fresh output buffer while
            # both references are alive, so mlx buffer donation cannot alias them).
            # Holding bare references relied on GatedDeltaNet's replace-not-mutate
            # behavior — an mlx implementation detail that does NOT hold across mlx
            # versions (0.30.6 replace vs 0.32.x buffer reuse): stale references
            # made rollback silently wrong and broke the losslessness guarantee.
            c0 = None if c.cache[0] is None else c.cache[0] + 0
            c1 = None if c.cache[1] is None else c.cache[1] + 0
            if c0 is not None or c1 is not None:
                mx.eval(*[x for x in (c0, c1) if x is not None])
            arr_snaps.append((i, c0, c1))
        elif isinstance(c, QSACache):
            qsa_snaps.append((i, c.kv.offset, c.idx.offset))

    ple_prev = model._ple_prev          # None or np.ndarray — replaced each step
    # deep-copy conv-state values too (same replace-not-mutate fragility class)
    ple_conv = {k: (v if v is None else v + 0) for k, v in model._ple_conv.items()}
    last_wide = model._last_hidden_wide  # mx.array ref — replaced each step
    if last_wide is not None:
        last_wide = last_wide + 0
        mx.eval(last_wide)

    return {
        "arr": arr_snaps,
        "qsa": qsa_snaps,
        "ple_prev": ple_prev,
        "ple_conv": ple_conv,
        "last_wide": last_wide,
    }


def _restore_cache(snap, cache, model):
    """Restore cache to the pre-batch-verify state captured by _snapshot_cache.

    For ArraysCache: swap back the old array references.
    For QSACache: restore the integer offsets (data in buffer is irrelevant past offset).
    For PLE and _last_hidden_wide: restore the old references.
    """
    from mlx_lm.models.cache import ArraysCache
    from mlx_qwen4exp.attention import QSACache

    for i, c0, c1 in snap["arr"]:
        cache[i].cache[0] = c0
        cache[i].cache[1] = c1

    for i, kv_off, idx_off in snap["qsa"]:
        cache[i].kv.offset = kv_off
        cache[i].idx.offset = idx_off

    model._ple_prev = snap["ple_prev"]
    model._ple_conv = snap["ple_conv"]
    model._last_hidden_wide = snap["last_wide"]


def generate_mtp_v2(model, tokenizer, model_dir, args, prompt_ids, max_tokens, label=""):
    """Speculative decode with k=1 MTP draft head — batch-verify fast path.

    Protocol (equality-safe):
      1. Prefill prompt → first_tok.
      2. Loop:
         a. MTP drafts draft_tok from _last_hidden_wide and cur_tok.
         b. Snapshot cache state (zero-cost reference snapshot).
         c. BATCH VERIFY: model([[cur_tok, draft_tok]], cache) → ver_logits [B, 2, vocab].
            accepted_tok = argmax(ver_logits[:, 0, :])   (identical to plain greedy)
         d. If draft_tok == accepted_tok (accept):
              Keep cache at offset+2. Emit accepted_tok.
              next_tok = argmax(ver_logits[:, 1, :]) — already computed. Emit it.
              seq_pos += 2.
         e. If draft rejected:
              Restore cache to snapshot (reference swap + offset rollback, ~100 ns).
              Run single-token verify: model([[cur_tok]], cache) → single_ver.
              accepted_tok = argmax(single_ver[:, 0, :]).
              Emit accepted_tok. seq_pos += 1.
              Roll back MTP cache offset.

    Equality guarantee:
      - ver_logits[:, 0, :] from T=2 batch == logits from T=1 single-token verify,
        because position 0 output under causal attention/recurrence depends only on
        prior context and cur_tok, NOT on draft_tok at position 1. This is true for
        both QSA (causal mask) and GatedDeltaNet (causal recurrence).
      - On rejection the fallback single-token verify is identical to plain greedy.
      - On acceptance next_tok from ver_logits[:, 1, :] is conditioned on draft_tok
        (now confirmed == accepted_tok), so it equals what plain greedy would produce
        at that position.

    Speedup mechanism:
      At ~75% acceptance: 1 batch-verify(T=2) + 1 draft → 2 tokens.
      verify(T=2) ≈ 1.05-1.15× verify(T=1) due to weight-bandwidth dominance.
      Average tokens per effective-verify-cycle > 1.5× → well above the 32 tok/s bar.
    """
    from mlx_qwen4exp.mtp import make_mtp_cache

    _force = getattr(mx, "eval")

    if model.mtp is None:
        raise RuntimeError(
            "generate_mtp_v2 called but model.mtp is None. "
            "Load the model with load_mtp=True."
        )

    eos = _eos_set(model_dir, args)
    inputs = mx.array(np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1))
    model.reset_ple_state()
    cache = model.make_cache()
    mtp_cache = make_mtp_cache()
    lm = model.model.language_model

    print(f"\n[gen_mtp_v2{(' '+label) if label else ''}] prompt_tokens={inputs.shape[1]}", flush=True)
    t0 = time.time()
    logits = model(inputs, cache=cache)
    tok_arr = mx.argmax(logits[:, -1, :], axis=-1)
    _force(tok_arr)
    prefill_dt = time.time() - t0
    prefill_tps = inputs.shape[1] / prefill_dt if prefill_dt > 0 else float("nan")

    out_ids: list = []
    t1 = time.time()
    n_tokens = 0
    n_drafts = 0
    n_accepted = 0
    n_batch_accepts = 0   # times we used the T=2 batch path (accepted)
    n_single_rejects = 0  # times we fell back to single-token verify (rejected)
    seq_pos: int = inputs.shape[1]

    first_tok = int(np.array(tok_arr)[0])
    if first_tok in eos:
        print("[eos]", flush=True)
    else:
        out_ids.append(first_tok)
        print(tokenizer.decode([first_tok]), end="", flush=True)
        n_tokens += 1

    while n_tokens < max_tokens:
        if not out_ids:
            break
        cur_tok = out_ids[-1]
        if cur_tok in eos:
            break

        # --- draft ---
        last_wide = model._last_hidden_wide[:, -1:, :, :]
        next_id = mx.array([[cur_tok]])
        mtp_kv_before = mtp_cache[0].kv.offset
        mtp_idx_before = mtp_cache[0].idx.offset
        draft_logits = model.mtp(
            last_wide, next_id,
            lm.embed_tokens, model.lm_head,
            cache=mtp_cache[0], pos_offset=seq_pos,
        )
        # async_eval draft: dispatches to GPU without blocking Python so we can
        # set up the batch-verify inputs while draft executes.
        mx.async_eval(draft_logits)
        n_drafts += 1

        # --- snapshot cache (reference-only, ~100 ns) ---
        snap = _snapshot_cache(cache, model)

        # Read draft_tok — this blocks until async_eval of draft_logits is done.
        draft_tok = int(np.array(mx.argmax(draft_logits[0, -1, :])))

        # --- batch verify: feed [cur_tok, draft_tok] in one forward ---
        two = mx.array([[cur_tok, draft_tok]])
        ver_logits = model(two, cache=cache)
        _force(ver_logits)
        accepted_tok = int(np.array(mx.argmax(ver_logits[0, 0, :])))

        if draft_tok == accepted_tok:
            # Accept: cache already advanced by 2, emit accepted_tok + next_tok.
            seq_pos += 2
            n_accepted += 1
            n_batch_accepts += 1

            if accepted_tok in eos:
                break
            out_ids.append(accepted_tok)
            print(tokenizer.decode([accepted_tok]), end="", flush=True)
            n_tokens += 1

            if n_tokens >= max_tokens:
                break

            next_tok = int(np.array(mx.argmax(ver_logits[0, 1, :])))
            if next_tok in eos:
                break
            out_ids.append(next_tok)
            print(tokenizer.decode([next_tok]), end="", flush=True)
            n_tokens += 1

        else:
            # Reject: restore state, fall back to single-token verify for equality.
            _restore_cache(snap, cache, model)
            mtp_cache[0].kv.offset = mtp_kv_before
            mtp_cache[0].idx.offset = mtp_idx_before
            n_single_rejects += 1

            one = mx.array([[cur_tok]])
            single_ver = model(one, cache=cache)
            _force(single_ver)
            accepted_tok = int(np.array(mx.argmax(single_ver[0, 0, :])))
            seq_pos += 1

            if accepted_tok in eos:
                break
            out_ids.append(accepted_tok)
            print(tokenizer.decode([accepted_tok]), end="", flush=True)
            n_tokens += 1

    decode_dt = time.time() - t1
    decode_tps = n_tokens / decode_dt if decode_dt > 0 else float("nan")
    accept_rate = n_accepted / n_drafts if n_drafts > 0 else float("nan")
    print(flush=True)
    text = tokenizer.decode(out_ids)
    print(
        f"[gen_mtp_v2{(' '+label) if label else ''}] "
        f"prefill {prefill_tps:.2f} tok/s ({inputs.shape[1]} tok {prefill_dt:.1f}s) | "
        f"decode {decode_tps:.2f} tok/s ({n_tokens} tok {decode_dt:.1f}s) | "
        f"accept_rate {accept_rate*100:.1f}% ({n_accepted}/{n_drafts}) | "
        f"batch_accept {n_batch_accepts} single_reject {n_single_rejects}",
        flush=True,
    )
    return text, decode_tps, accept_rate


def equality_test(model, tokenizer, model_dir, args, use_v2: bool = True) -> bool:
    """Leg 3 equality test: 200-token greedy on 3 prompts, MTP must == non-MTP.

    Tests generate_mtp_v2 (batch-verify fast path) by default.
    Set use_v2=False to fall back to testing the original sequential generate_mtp.

    Returns True iff ALL prompts produce identical token sequences.
    This is the constructed-equivalence leg — a single mismatch is a bug.
    """
    PROMPTS = [
        "The capital of France is",
        "Machine learning is a subset of artificial intelligence that",
        "In the beginning God created the heavens and the earth",
    ]
    N_TOKENS = 200
    all_pass = True
    mtp_fn = generate_mtp_v2 if use_v2 else generate_mtp
    variant = "v2" if use_v2 else "v1"

    for i, prompt in enumerate(PROMPTS):
        prompt_ids = tokenizer(prompt, return_tensors="np")["input_ids"].reshape(-1)

        # non-MTP greedy
        text_plain = generate(model, tokenizer, model_dir, args, prompt_ids, N_TOKENS, label=f"eq-plain-{i}")
        ids_plain = tokenizer(text_plain, return_tensors="np")["input_ids"].reshape(-1).tolist()

        # MTP greedy (v1 or v2)
        text_mtp, _, _ = mtp_fn(model, tokenizer, model_dir, args, prompt_ids, N_TOKENS, label=f"eq-mtp-{variant}-{i}")
        ids_mtp = tokenizer(text_mtp, return_tensors="np")["input_ids"].reshape(-1).tolist()

        if ids_plain == ids_mtp:
            print(f"[equality-{variant}] prompt {i}: PASS ({len(ids_plain)} tokens identical)", flush=True)
        else:
            # find first mismatch
            mismatch = next(
                (j for j, (a, b) in enumerate(zip(ids_plain, ids_mtp)) if a != b),
                min(len(ids_plain), len(ids_mtp)),
            )
            print(
                f"[equality-{variant}] prompt {i}: FAIL — first mismatch at token {mismatch} "
                f"(plain={ids_plain[mismatch] if mismatch < len(ids_plain) else 'EOL'} "
                f"mtp={ids_mtp[mismatch] if mismatch < len(ids_mtp) else 'EOL'})",
                flush=True,
            )
            all_pass = False

    return all_pass


def benchmark(model, tokenizer, model_dir, args) -> None:
    """Compare plain / MTP-v1 / MTP-v2 on the 3 standard prompts, 200 tokens each.

    Prints a summary table of decode tok/s and acceptance rates.
    """
    PROMPTS = [
        "The capital of France is",
        "Machine learning is a subset of artificial intelligence that",
        "In the beginning God created the heavens and the earth",
    ]
    N_TOKENS = 200

    results = []
    for i, prompt in enumerate(PROMPTS):
        prompt_ids = tokenizer(prompt, return_tensors="np")["input_ids"].reshape(-1)

        t0 = time.time()
        generate(model, tokenizer, model_dir, args, prompt_ids, N_TOKENS, label=f"bench-plain-{i}")
        # re-run to get measured tps (generate() doesn't return it)
        model.reset_ple_state()
        cache = model.make_cache()
        inputs = mx.array(np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1))
        logits = model(inputs, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        t_start = time.time()
        n = 0
        out_tok = int(np.array(tok)[0])
        eos = _eos_set(model_dir, args)
        while n < N_TOKENS and out_tok not in eos:
            y = mx.array([[out_tok]])
            logits = model(y, cache=cache)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            out_tok = int(np.array(tok)[0])
            n += 1
        plain_tps = n / (time.time() - t_start) if n else float("nan")

        _, v1_tps, v1_acc = generate_mtp(
            model, tokenizer, model_dir, args, prompt_ids, N_TOKENS, label=f"bench-mtp1-{i}"
        )
        _, v2_tps, v2_acc = generate_mtp_v2(
            model, tokenizer, model_dir, args, prompt_ids, N_TOKENS, label=f"bench-mtp2-{i}"
        )
        results.append((i, plain_tps, v1_tps, v1_acc, v2_tps, v2_acc))

    print("\n=== BENCHMARK SUMMARY ===", flush=True)
    print(f"{'Prompt':>6}  {'plain tok/s':>12}  {'MTP-v1 tok/s':>13}  {'v1 acc%':>8}  {'MTP-v2 tok/s':>13}  {'v2 acc%':>8}", flush=True)
    for (i, pt, v1t, v1a, v2t, v2a) in results:
        print(
            f"  P{i}    {pt:12.2f}  {v1t:13.2f}  {v1a*100:8.1f}  {v2t:13.2f}  {v2a*100:8.1f}",
            flush=True,
        )
    avg_plain = sum(r[1] for r in results) / len(results)
    avg_v1 = sum(r[2] for r in results) / len(results)
    avg_v2 = sum(r[4] for r in results) / len(results)
    print(
        f"  avg   {avg_plain:12.2f}  {avg_v1:13.2f}  {'':>8}  {avg_v2:13.2f}",
        flush=True,
    )
    print(f"\n  v2 speedup over plain: {avg_v2/avg_plain:.2f}×", flush=True)
    print(f"  v2 speedup over v1:    {avg_v2/avg_v1:.2f}×", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chat", action="store_true", help="apply the chat template (thinking off)")
    ap.add_argument("--probe", action="store_true", help="per-layer norm probe, then exit")
    ap.add_argument("--top5", action="store_true", help="print top-5 next tokens after prefill")
    ap.add_argument("--mtp", action="store_true", help="use MTP speculative decoding (v1, sequential)")
    ap.add_argument("--mtp-v2", action="store_true", help="use MTP v2 batch-verify speculative decoding")
    ap.add_argument(
        "--equality-test", action="store_true",
        help="run Leg 3: 200-token greedy on 3 prompts, verify MTP-v2==non-MTP (token-for-token)"
    )
    ap.add_argument(
        "--equality-test-v1", action="store_true",
        help="run Leg 3 against original sequential MTP (v1) for comparison"
    )
    ap.add_argument(
        "--benchmark", action="store_true",
        help="compare plain / MTP-v1 / MTP-v2 on 3 standard prompts, 200 tokens each"
    )
    args_cli = ap.parse_args(argv)

    model_dir_str = args_cli.model_dir or os.environ.get("QWEN_MODEL_DIR")
    if not model_dir_str:
        ap.error("--model-dir is required (or set QWEN_MODEL_DIR env var)")
    model_dir = Path(model_dir_str)
    stage("start")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    stage("tokenizer loaded")

    model = load_model(model_dir)
    attach_ple(model, model_dir)
    stage("model ready")

    args = model.args

    if args_cli.probe:
        print("\n=== PROBE (per-layer norms on 'The capital of France is') ===", flush=True)
        probe(model, tokenizer, model_dir, args)
        return 0

    if args_cli.top5:
        top5(model, tokenizer, model_dir, args)

    if args_cli.benchmark:
        print("\n=== BENCHMARK (plain / MTP-v1 / MTP-v2, 200 tok, 3 prompts) ===", flush=True)
        benchmark(model, tokenizer, model_dir, args)
        stage("benchmark done")
        return 0

    if args_cli.equality_test:
        print("\n=== EQUALITY TEST v2 (Leg 3: MTP-v2 == non-MTP, 200 tokens, 3 prompts) ===", flush=True)
        passed = equality_test(model, tokenizer, model_dir, args, use_v2=True)
        print(f"\n[equality-v2] {'ALL PASS' if passed else 'FAILED'}", flush=True)
        stage("equality test done")
        return 0 if passed else 1

    if args_cli.equality_test_v1:
        print("\n=== EQUALITY TEST v1 (Leg 3: MTP-v1 == non-MTP, 200 tokens, 3 prompts) ===", flush=True)
        passed = equality_test(model, tokenizer, model_dir, args, use_v2=False)
        print(f"\n[equality-v1] {'ALL PASS' if passed else 'FAILED'}", flush=True)
        stage("equality test done")
        return 0 if passed else 1

    if args_cli.chat:
        msgs = [{"role": "user", "content": args_cli.prompt}]
        try:
            enc = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        # apply_chat_template may return a plain list[int] or a BatchEncoding dict.
        if isinstance(enc, dict) or hasattr(enc, "input_ids"):
            ids = enc["input_ids"]
        else:
            ids = enc
        prompt_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        print(f"[chat] rendered prompt ({len(prompt_ids)} tok):", flush=True)
        print(repr(tokenizer.decode(prompt_ids.tolist())), flush=True)
    else:
        prompt_ids = tokenizer(args_cli.prompt, return_tensors="np")["input_ids"].reshape(-1)

    if args_cli.mtp_v2:
        text, tps, accept_rate = generate_mtp_v2(
            model, tokenizer, model_dir, args, prompt_ids, args_cli.max_tokens,
            label="chat" if args_cli.chat else "plain",
        )
    elif args_cli.mtp:
        text, tps, accept_rate = generate_mtp(
            model, tokenizer, model_dir, args, prompt_ids, args_cli.max_tokens,
            label="chat" if args_cli.chat else "plain",
        )
    else:
        text = generate(
            model, tokenizer, model_dir, args, prompt_ids, args_cli.max_tokens,
            label="chat" if args_cli.chat else "plain",
        )
    print("\n=== GENERATED ===", flush=True)
    print(text, flush=True)
    stage("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
