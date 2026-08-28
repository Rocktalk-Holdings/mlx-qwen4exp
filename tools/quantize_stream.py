#!/usr/bin/env python3
"""Shard-streaming 4-bit quantizer for a converted Qwen3.8-Flash-Next MLX checkpoint.

Why streaming: a bf16 eager load (~234 GB) may not coexist with other large model
servers on the same machine. This quantizer never holds more than one ~5 GB shard in
RAM. It writes an mlx_lm-convention 4-bit checkpoint (~65-70 GB resident).

Convention (mirrors mlx_lm.utils.quantize_model exactly):
  * A module is quantizable iff it has `to_quantized` AND weight.shape[-1] % group_size == 0.
  * Quantization policy is driven by Model.quant_predicate: norms / gammas / A_log /
    dt_bias are skipped; routing gates (mlp.gate, shared_expert_gate) go to 8-bit g64;
    embed_tokens + lm_head go to 8-bit g64 (matches the reference GGUF Q8_0 layers);
    everything else 4-bit g64.
  * Quantizing a leaf `X.weight` emits three tensors: `X.weight` (packed uint32),
    `X.scales`, `X.biases` -- the exact names QuantizedLinear / QuantizedSwitchLinear /
    QuantizedEmbedding expose, so nn.quantize(model, class_predicate=...) at load time
    reconstructs a matching tree.
  * config.json gets `config["quantization"] = {group_size, bits, mode, <path>: {...}}`
    per mlx_lm convention.

The 95 GB fp16 ngram_table.bin is NOT quantized: it is gathered per-token off a memmap.
A symlink is created in the output dir pointing at the source table (no copy).

Usage:
    python3 tools/quantize_stream.py --src-dir /path/to/MLX-bf16 --dst-dir /path/to/MLX-4bit
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mlx_qwen4exp.config import ModelArgs
from mlx_qwen4exp.model import Model

GROUP_SIZE = 64
BITS = 4
RSS_ABORT_GB = 40.0

# files to copy verbatim (config.json is written separately with the quant section)
COPY_FILES = [
    "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
    "chat_template.jinja", "generation_config.json", "special_tokens_map.json",
    "ple_constants.json", "added_tokens.json",
]


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def vm_free_gb() -> float:
    import subprocess
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


def build_quant_map(args: ModelArgs):
    """Return {module_path: {'group_size','bits'}} for every module the model's own
    quant_predicate would quantize, honouring the g64-divisibility guard that
    nn.quantize applies. Built on an EMPTY model (cheap; no weights loaded)."""
    m = Model(args)
    pred = m.quant_predicate
    qmap = {}

    def wrapped(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % GROUP_SIZE != 0:
            return False
        r = pred(path, module)
        if isinstance(r, dict):
            qmap[path] = {"group_size": r.get("group_size", GROUP_SIZE), "bits": r.get("bits", BITS)}
            return r
        if r:
            qmap[path] = {"group_size": GROUP_SIZE, "bits": BITS}
        return r

    nn.quantize(m, GROUP_SIZE, BITS, class_predicate=wrapped)
    del m
    gc.collect()
    return qmap


def quantize_shard(weights: dict, qmap: dict) -> dict:
    """Quantize the quantizable tensors in one shard's weight dict. Returns a new dict
    with weight/scales/biases triples for quantized leaves and pass-through for the rest."""
    out = {}
    for k, w in weights.items():
        module_path = k[:-len(".weight")] if k.endswith(".weight") else None
        params = qmap.get(module_path) if module_path is not None else None
        if params is None:
            out[k] = w
            continue
        wq, scales, biases = mx.quantize(w, group_size=params["group_size"], bits=params["bits"])
        out[k] = wq
        out[f"{module_path}.scales"] = scales
        out[f"{module_path}.biases"] = biases
    return out


def write_config(src: Path, dst: Path, qmap: dict) -> None:
    cfg = json.load(open(src / "config.json"))
    quant = {"group_size": GROUP_SIZE, "bits": BITS, "mode": "affine"}
    # per-path overrides only where they differ from the top-level defaults
    for path, p in qmap.items():
        if p["group_size"] != GROUP_SIZE or p["bits"] != BITS:
            quant[path] = {"group_size": p["group_size"], "bits": p["bits"], "mode": "affine"}
    cfg["quantization"] = quant
    cfg["quantization_config"] = quant
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)
    n_over = sum(1 for k in quant if isinstance(quant[k], dict))
    print(f"[cfg] wrote config.json with quantization ({n_over} per-path overrides)", flush=True)


def build_index(dst: Path) -> None:
    """Write model.safetensors.index.json mapping every tensor -> its shard file."""
    weight_map = {}
    total = 0
    for sh in sorted(dst.glob("model-*-of-*.safetensors")):
        w = mx.load(str(sh))
        for k in w.keys():
            weight_map[k] = sh.name
        total += sh.stat().st_size
        del w
    idx = {"metadata": {"total_size": total}, "weight_map": weight_map}
    json.dump(idx, open(dst / "model.safetensors.index.json", "w"), indent=2)
    print(f"[index] wrote index.json ({len(weight_map)} tensors)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Shard-streaming 4-bit quantizer for Qwen3.8-Flash-Next MLX checkpoint"
    )
    ap.add_argument("--src-dir", required=True, type=Path,
                    help="Path to the bf16 MLX checkpoint produced by convert.py")
    ap.add_argument("--dst-dir", required=True, type=Path,
                    help="Output path for the 4-bit quantized checkpoint")
    args_ns = ap.parse_args(argv)
    src: Path = args_ns.src_dir
    dst: Path = args_ns.dst_dir

    if not src.is_dir():
        print(f"ERROR: --src-dir {src} not found", file=sys.stderr)
        return 2

    t_all = time.time()
    free = vm_free_gb()
    print(f"[start] src={src}  dst={dst}", flush=True)
    print(f"[start] free+inactive={free:.1f} GB  RSS={rss_gb():.1f} GB", flush=True)

    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(src / "config.json"))
    args = ModelArgs.from_dict(cfg)
    print("[qmap] building quant map on empty model ...", flush=True)
    qmap = build_quant_map(args)
    n4 = sum(1 for v in qmap.values() if v["bits"] == 4)
    n8 = sum(1 for v in qmap.values() if v["bits"] == 8)
    print(f"[qmap] {len(qmap)} quantizable modules ({n4} @4bit, {n8} @8bit); RSS={rss_gb():.1f} GB", flush=True)

    shards = sorted(src.glob("model-*-of-*.safetensors"))
    n = len(shards)
    print(f"[quant] {n} shards", flush=True)
    peak_rss = rss_gb()
    for i, sh in enumerate(shards, 1):
        w = mx.load(str(sh))
        qw = quantize_shard(w, qmap)
        mx.eval({"done": mx.array(1)})  # force graph flush after quantize
        out_path = dst / sh.name
        mx.save_safetensors(str(out_path), qw)
        del w, qw
        gc.collect()
        peak_rss = max(peak_rss, rss_gb())
        if i % 5 == 0 or i == n:
            print(f"[quant] shard {i}/{n} -> {sh.name}  RSS={rss_gb():.1f} GB (peak {peak_rss:.1f})  free={vm_free_gb():.1f} GB", flush=True)
        if rss_gb() > RSS_ABORT_GB:
            raise MemoryError(f"RSS {rss_gb():.1f} GB exceeded {RSS_ABORT_GB} GB abort threshold at shard {i}")

    build_index(dst)
    write_config(src, dst, qmap)

    for f in COPY_FILES:
        s = src / f
        if s.exists():
            shutil.copy2(s, dst / f)
            print(f"[copy] {f}", flush=True)

    # symlink the fp16 ngram table into the output dir (do not copy 95 GB)
    link = dst / "ngram_table.bin"
    if link.exists() or link.is_symlink():
        link.unlink()
    table_src = src / "ngram_table.bin"
    if table_src.exists():
        os.symlink(str(table_src), str(link))
        print(f"[link] ngram_table.bin -> {table_src}", flush=True)
    else:
        print("[warn] ngram_table.bin not found in src -- skipping symlink", flush=True)

    os.sync()
    print(f"[done] quantize wall time {time.time()-t_all:.1f}s  peak RSS {peak_rss:.1f} GB", flush=True)
    out_shards = sorted(dst.glob("model-*-of-*.safetensors"))
    total = sum(p.stat().st_size for p in out_shards)
    print(f"[verify] {len(out_shards)} output shards, {total/1e9:.1f} GB total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
