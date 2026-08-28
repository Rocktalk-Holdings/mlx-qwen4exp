#!/usr/bin/env python3.11
"""Convert a raw HF Qwen3.8-Flash-Next (qwen4_exp) checkpoint into an MLX package.

What it produces in ``--out-dir``:
  * ``config.json``                — a copy of the HF config (unchanged).
  * ``model-000NN-of-000MM.safetensors`` — the transformer weights, sanitized by
    ``Model.sanitize`` and re-sharded to ~5 GB each (mx.save_safetensors).
  * ``model.safetensors.index.json`` — the standard weight_map so mlx_lm can load it.
  * ``ple_constants.json``          — the three I64 hash constants (layer_multipliers,
    ngram_heads_offsets, ngram_heads_vocab_sizes) plus table geometry. Loaded at runtime
    onto ``Model._ple_multipliers / _ple_offsets / _ple_vocab_sizes``.
  * ``ngram_table.bin``             — the 128 n-gram shards concatenated into ONE
    memory-mapped float16 file, shape [320001536, 160] = 51.2B params (~102 GB). Built by
    streaming: never more than one shard is resident in RAM. Skipped with ``--skip-table``.

Design notes / landmines:
  * The download may be incomplete and there is often NO model.safetensors.index.json. This
    script therefore SCANS the shard files directly (mx.core.load per file) and does not
    depend on the index. If the index exists it is used only to sanity-check completeness.
  * ``Model.sanitize`` is applied PROGRESSIVELY, per shard, so we never hold the whole
    355 GB checkpoint in memory. Because the fused-expert split and the indexer split need
    ALL of a layer's related keys present together, we buffer per-layer expert/indexer
    fragments until complete, then sanitize+flush them. Non-fused keys are sanitized and
    emitted immediately.
  * RESTARTABLE / IDEMPOTENT: every output file is written to a ``.tmp`` sibling and
    atomically renamed; a completed output is skipped on re-run. The table build resumes by
    checking which shard rows are already non-zero is NOT reliable, so the table is written
    to ``ngram_table.bin.partial`` with a companion ``ngram_table.progress`` marker listing
    finished shard indices; on restart it resumes from the first unfinished shard.

DO NOT expect to run the full 355 GB conversion casually — the table alone is ~102 GB and
the transformer another ~15 GB. Smoke-test the plumbing with ``--limit-shards N`` (loads
only the first N transformer shards, strict=False) before the real run.

Usage:
    python3.11 -m mlx_qwen4exp.convert --hf-dir DIR --out-dir OUT [--skip-table]
                                       [--limit-shards N] [--shard-size-gb 5]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# allow `python3.11 mlx_qwen4exp/convert.py ...` as well as `-m mlx_qwen4exp.convert`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_qwen4exp.config import ModelArgs
    from mlx_qwen4exp.model import Model
else:
    from .config import ModelArgs
    from .model import Model

import mlx.core as mx


# --------------------------------------------------------------------------- shard discovery
_SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors$")


def discover_transformer_shards(hf_dir: Path) -> List[Path]:
    """Return the transformer safetensors shards present, in shard-index order.

    Independent of model.safetensors.index.json (which may be absent mid-download)."""
    shards = []
    for p in hf_dir.glob("model-*-of-*.safetensors"):
        m = _SHARD_RE.search(p.name)
        if m:
            shards.append((int(m.group(1)), p))
    shards.sort(key=lambda t: t[0])
    return [p for _, p in shards]


def expected_shard_count(hf_dir: Path) -> Optional[int]:
    for p in hf_dir.glob("model-*-of-*.safetensors"):
        m = _SHARD_RE.search(p.name)
        if m:
            return int(m.group(2))
    return None


# --------------------------------------------------------------------------- config
def load_args(hf_dir: Path) -> Tuple[ModelArgs, dict]:
    with open(hf_dir / "config.json") as f:
        cfg = json.load(f)
    return ModelArgs.from_dict(cfg), cfg


# --------------------------------------------------------------------------- PLE constants
_PLE_CONST_SUFFIXES = {
    "layer_multipliers": "ple_embedding.layer_multipliers",
    "ngram_heads_offsets": "ple_embedding.ngram_heads_offsets",
    "ngram_heads_vocab_sizes": "ple_embedding.ngram_heads_vocab_sizes",
}


def extract_ple_constants(hf_dir: Path, args: ModelArgs) -> Dict[str, List[int]]:
    """Read the three I64 PLE hash constants EXACTLY (int64, no float rounding).

    They are tiny 1-D tensors; scan shards until all three are found. mx keeps int64 as
    int64, so ``np.array(v)`` preserves the 45-bit multipliers bit-for-bit.
    """
    found: Dict[str, List[int]] = {}
    for shard in discover_transformer_shards(hf_dir):
        w = mx.load(str(shard))
        for name, suffix in _PLE_CONST_SUFFIXES.items():
            if name in found:
                continue
            for k, v in w.items():
                if k.endswith(suffix):
                    arr = np.array(v)
                    found[name] = [int(x) for x in arr.reshape(-1).tolist()]
                    break
        if len(found) == len(_PLE_CONST_SUFFIXES):
            break
    missing = set(_PLE_CONST_SUFFIXES) - set(found)
    if missing:
        raise ValueError(f"PLE hash constants not found in checkpoint: {sorted(missing)}")
    return found


def write_ple_constants(out_dir: Path, consts: Dict[str, List[int]], args: ModelArgs) -> Path:
    payload = {
        "layer_multipliers": consts["layer_multipliers"],
        "ngram_heads_offsets": consts["ngram_heads_offsets"],
        "ngram_heads_vocab_sizes": consts["ngram_heads_vocab_sizes"],
        "eos_token_id": args.eos_token_id,
        "ngram_size": args.ngram_size,
        "heads_per_ngram": args.heads_per_ngram,
        "ple_n_heads": args.ple_n_heads,
        "ple_head_dim": args.ple_head_dim,
        "table_file": "ngram_table.bin",
        "table_dtype": "float16",
    }
    path = out_dir / "ple_constants.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- ngram table
_SHARD_TENSOR_RE = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")


def find_ngram_shards(hf_dir: Path) -> Dict[int, Path]:
    """Map ngram shard index -> the safetensors file that holds it (one tensor per file
    at these sizes, but we do not assume that)."""
    out: Dict[int, Path] = {}
    for shard_file in discover_transformer_shards(hf_dir):
        # peek at names cheaply via mx.load (lazy header) — mx.load reads the whole file,
        # so cache the mapping across constants+shards passes at the call site if needed.
        w = mx.load(str(shard_file))
        for k in w.keys():
            m = _SHARD_TENSOR_RE.search(k)
            if m:
                out[int(m.group(1))] = shard_file
    return out


def build_ngram_table(
    hf_dir: Path,
    out_dir: Path,
    args: ModelArgs,
) -> Path:
    """Concatenate the 128 ngram shards into one memmapped float16 [total_rows, row_dim].

    Streaming + restartable: writes to ``ngram_table.bin.partial`` guided by a
    ``ngram_table.progress`` marker (JSON: finished shard indices + geometry). Only one
    shard is resident at a time. On success renames to ``ngram_table.bin``.
    """
    final_path = out_dir / "ngram_table.bin"
    if final_path.exists():
        print(f"[table] {final_path.name} already exists — skipping (idempotent).")
        return final_path

    n_parts = args.split_ngram_parts
    shard_map = find_ngram_shards(hf_dir)
    have = sorted(shard_map)
    if len(have) < n_parts:
        raise ValueError(
            f"only {len(have)}/{n_parts} ngram shards present; table build needs all of "
            f"them (download incomplete). Re-run with --skip-table or wait for the "
            f"remaining shards. Missing: {sorted(set(range(n_parts)) - set(have))[:8]}..."
        )

    # determine geometry from shard 0 (all but possibly the last are uniform)
    row_dim = None
    shard_rows: Dict[int, int] = {}
    for idx in have:
        w = mx.load(str(shard_map[idx]))
        for k, v in w.items():
            m = _SHARD_TENSOR_RE.search(k)
            if m and int(m.group(1)) == idx:
                shard_rows[idx] = int(v.shape[0])
                rd = int(v.shape[1])
                row_dim = rd if row_dim is None else row_dim
                if rd != row_dim:
                    raise ValueError(f"shard {idx} row_dim {rd} != {row_dim}")
                break
    total_rows = sum(shard_rows[i] for i in range(n_parts))
    # cumulative row offset for each shard (shards may have a short final one)
    offsets = {}
    acc = 0
    for i in range(n_parts):
        offsets[i] = acc
        acc += shard_rows[i]

    partial = out_dir / "ngram_table.bin.partial"
    progress_path = out_dir / "ngram_table.progress"

    finished: set = set()
    if progress_path.exists() and partial.exists():
        try:
            prog = json.load(open(progress_path))
            if (
                prog.get("total_rows") == total_rows
                and prog.get("row_dim") == row_dim
                and partial.stat().st_size == total_rows * row_dim * 2
            ):
                finished = set(prog.get("finished", []))
                print(f"[table] resuming; {len(finished)}/{n_parts} shards already written.")
            else:
                print("[table] progress marker mismatched geometry — restarting table.")
                partial.unlink(missing_ok=True)
        except Exception:
            partial.unlink(missing_ok=True)

    # open (or create) the memmap at full size
    mode = "r+" if partial.exists() else "w+"
    table = np.memmap(partial, dtype=np.float16, mode=mode, shape=(total_rows, row_dim))

    for idx in range(n_parts):
        if idx in finished:
            continue
        w = mx.load(str(shard_map[idx]))
        shard_arr = None
        for k, v in w.items():
            m = _SHARD_TENSOR_RE.search(k)
            if m and int(m.group(1)) == idx:
                shard_arr = np.array(v.astype(mx.float16))
                break
        if shard_arr is None:
            raise ValueError(f"shard {idx} tensor vanished from {shard_map[idx]}")
        start = offsets[idx]
        table[start : start + shard_arr.shape[0]] = shard_arr
        del shard_arr, w
        table.flush()
        finished.add(idx)
        tmp_prog = progress_path.with_suffix(".progress.tmp")
        with open(tmp_prog, "w") as f:
            json.dump(
                {"total_rows": total_rows, "row_dim": row_dim, "finished": sorted(finished)},
                f,
            )
        os.replace(tmp_prog, progress_path)
        print(f"[table] wrote shard {idx + 1}/{n_parts} (rows {start}..{start + shard_rows[idx]})")

    table.flush()
    del table
    # paranoia (Drew's law): sync, verify size, then atomically rename
    _sync()
    want = total_rows * row_dim * 2
    got = partial.stat().st_size
    if got != want:
        raise ValueError(f"table size {got} != expected {want}; NOT renaming")
    os.replace(partial, final_path)
    progress_path.unlink(missing_ok=True)
    _sync()
    print(f"[table] DONE: {final_path} [{total_rows}, {row_dim}] float16 ({got/1e9:.1f} GB)")
    return final_path


def _sync() -> None:
    try:
        os.sync()
    except AttributeError:
        pass


# --------------------------------------------------------------------------- transformer weights
def _is_ple_junk(k: str) -> bool:
    return ".ple.ple_embedding." in k


def _is_visual_or_mtp(k: str) -> bool:
    return k.startswith("model.visual.") or k.startswith("mtp.") or k.startswith("model.mtp.")


def stream_and_sanitize_transformer(
    hf_dir: Path,
    model: Model,
    limit_shards: Optional[int],
) -> Dict[str, mx.array]:
    """Load every transformer shard, drop junk + PLE-table keys, and run Model.sanitize.

    Model.sanitize buffers per-layer fused/indexer keys internally (it processes the whole
    dict), so the simplest correct approach at this scale is: collect the NON-junk,
    NON-table keys from every shard into one dict, then sanitize once. At real dims that
    dict is ~15 GB (bf16 transformer minus the 102 GB table), which fits Drew's 512 GB box.

    A per-shard incremental sanitize is possible but the expert-split needs all of a
    layer's expert fragments (they are one fused tensor here, so already co-located), so a
    single pass is both correct and simplest. limit_shards caps the pass for smoke tests.
    """
    collected: Dict[str, mx.array] = {}
    shards = discover_transformer_shards(hf_dir)
    if limit_shards is not None:
        shards = shards[:limit_shards]
    for i, shard in enumerate(shards):
        w = mx.load(str(shard))
        kept = 0
        for k, v in w.items():
            if _is_visual_or_mtp(k) or _is_ple_junk(k):
                continue
            collected[k] = v
            kept += 1
        print(f"[xf] shard {i + 1}/{len(shards)} {shard.name}: kept {kept}/{len(w)} keys")
        del w
    print(f"[xf] collected {len(collected)} transformer keys; sanitizing...")
    return model.sanitize(collected)


def resave_transformer(
    sanitized: Dict[str, mx.array],
    out_dir: Path,
    shard_size_gb: float,
) -> None:
    """Re-shard the sanitized weights into ~shard_size_gb safetensors files and write the
    standard index.json weight_map. Idempotent: skips if index.json already present."""
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.exists():
        print(f"[xf] {index_path.name} exists — skipping re-save (idempotent).")
        return

    items = sorted(sanitized.items())
    max_bytes = int(shard_size_gb * (1024 ** 3))

    # bucket keys into shards by cumulative byte size
    buckets: List[List[str]] = [[]]
    cur = 0
    for k, v in items:
        nbytes = v.nbytes
        if cur > 0 and cur + nbytes > max_bytes:
            buckets.append([])
            cur = 0
        buckets[-1].append(k)
        cur += nbytes

    n = len(buckets)
    weight_map: Dict[str, str] = {}
    total_size = 0
    for bi, keys in enumerate(buckets):
        fname = f"model-{bi + 1:05d}-of-{n:05d}.safetensors"
        # mx.save_safetensors force-appends ".safetensors" if absent, so a plain ".tmp"
        # name becomes "*.tmp.safetensors". Write to an explicit temp *.safetensors and
        # atomically rename to the final name.
        tmp = out_dir / f".tmp-{fname}"
        shard_w = {k: sanitized[k] for k in keys}
        for k in keys:
            total_size += sanitized[k].nbytes
            weight_map[k] = fname
        mx.save_safetensors(str(tmp), shard_w)
        os.replace(tmp, out_dir / fname)
        print(f"[xf] saved {fname} ({len(keys)} tensors)")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    tmp_idx = index_path.with_suffix(".json.tmp")
    with open(tmp_idx, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp_idx, index_path)
    _sync()
    print(f"[xf] wrote {index_path.name} ({len(weight_map)} tensors across {n} shards)")


def copy_config(hf_dir: Path, out_dir: Path) -> None:
    for name in ("config.json", "generation_config.json", "chat_template.jinja",
                 "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = hf_dir / name
        if src.exists():
            dst = out_dir / name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            print(f"[cfg] copied {name}")


# --------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert HF qwen4_exp -> MLX package")
    ap.add_argument("--hf-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skip-table", action="store_true",
                    help="skip the 102 GB ngram table build (download may be incomplete)")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="smoke test: only load the first N transformer shards (strict=False)")
    ap.add_argument("--shard-size-gb", type=float, default=5.0)
    args_ns = ap.parse_args(argv)

    hf_dir: Path = args_ns.hf_dir
    out_dir: Path = args_ns.out_dir
    if not hf_dir.is_dir():
        print(f"ERROR: --hf-dir {hf_dir} not found", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    args, _cfg = load_args(hf_dir)
    print(f"[cfg] {args.num_hidden_layers} layers, hidden {args.hidden_size}, "
          f"vocab {args.vocab_size}, {args.num_experts} experts")

    exp = expected_shard_count(hf_dir)
    have = len(discover_transformer_shards(hf_dir))
    if exp is not None and have < exp:
        print(f"[warn] {have}/{exp} transformer shards present — checkpoint download may be "
              f"incomplete. Proceeding (strict=False for missing keys).")

    copy_config(hf_dir, out_dir)

    # PLE constants (tiny, always safe to extract if present)
    try:
        consts = extract_ple_constants(hf_dir, args)
        write_ple_constants(out_dir, consts, args)
        print(f"[ple] wrote ple_constants.json "
              f"(mult[0]={consts['layer_multipliers'][0]}, "
              f"{len(consts['ngram_heads_offsets'])} heads)")
    except ValueError as e:
        print(f"[ple] constants not extracted: {e}")

    # transformer weights
    model = Model(args)
    sanitized = stream_and_sanitize_transformer(hf_dir, model, args_ns.limit_shards)

    strict = args_ns.limit_shards is None
    # report unmatched keys BOTH directions (unmatched = bug)
    model_keys = set(_flatten_param_names(model))
    produced = set(sanitized.keys())
    missing_in_produced = model_keys - produced
    extra_in_produced = produced - model_keys
    if extra_in_produced:
        print(f"[xf][WARN] {len(extra_in_produced)} produced keys NOT in model (bug?):")
        for k in sorted(extra_in_produced)[:20]:
            print("    +", k)
    if missing_in_produced:
        tag = "WARN" if strict else "info(limit)"
        print(f"[xf][{tag}] {len(missing_in_produced)} model params NOT produced:")
        for k in sorted(missing_in_produced)[:20]:
            print("    -", k)

    try:
        model.load_weights(list(sanitized.items()), strict=strict)
        print(f"[xf] load_weights OK (strict={strict})")
    except ValueError as e:
        if strict:
            print(f"[xf][ERROR] strict load failed: {e}", file=sys.stderr)
            return 1
        print(f"[xf][info] non-strict load: {e}")

    resave_transformer(sanitized, out_dir, args_ns.shard_size_gb)

    # ngram table
    if args_ns.skip_table:
        print("[table] --skip-table set; not building ngram_table.bin.")
    else:
        build_ngram_table(hf_dir, out_dir, args)

    print("[done] conversion complete ->", out_dir)
    return 0


def _flatten_param_names(model: Model) -> Iterable[str]:
    from mlx.utils import tree_flatten

    return [k for k, _ in tree_flatten(model.parameters())]


if __name__ == "__main__":
    raise SystemExit(main())
