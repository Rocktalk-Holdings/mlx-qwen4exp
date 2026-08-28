#!/usr/bin/env python3
"""Quantize and inject the mtp.* tensors into an existing 4-bit checkpoint.

The 49 main-model shards are NOT touched. This script:
  1. Reads each BF16 shard that contains mtp.* tensors.
  2. Applies sanitize_mtp_weights to transform them to module-tree keys.
  3. Quantizes the quantizable ones using the same predicates as the main model.
  4. Writes a new shard mtp-weights.safetensors in the 4-bit directory.
  5. Patches model.safetensors.index.json to include the mtp keys.
  6. Patches config.json with load_mtp=true.

Memory: loads one BF16 shard at a time (~5 GB). Peak RSS well under 40 GB.

Usage:
    python3 tools/quantize_mtp.py \
        --bf16-dir /path/to/BF16-checkpoint \
        --dst-dir  /path/to/MLX-4bit-checkpoint \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
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
from mlx_qwen4exp.mtp import sanitize_mtp_weights

MTP_SHARD_NAME = "mtp-weights.safetensors"
GROUP_SIZE = 64
BITS = 4


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def vm_free_gb() -> float:
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    page = 16384
    free = inactive = 0
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split("bytes")[0].strip())
        elif line.startswith("Pages free:"):
            free = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages inactive:"):
            inactive = int(line.split(":")[1].strip().rstrip("."))
    return (free + inactive) * page / 1e9


def stage(msg: str) -> None:
    print(f"[{msg:<40}] RSS={rss_gb():.1f}GB free={vm_free_gb():.1f}GB", flush=True)


def build_mtp_quant_predicate(args: ModelArgs):
    """Return a predicate for mtp.* weights matching the main model's quant_predicate."""
    empty_model = Model(ModelArgs(load_mtp=True))
    pred = empty_model.quant_predicate
    del empty_model
    gc.collect()

    def mtp_pred(key_no_prefix: str, w: mx.array) -> bool | dict:
        """key_no_prefix: module-relative key WITHOUT the mtp. prefix."""
        if key_no_prefix.endswith(".weight"):
            module_path = "mtp." + key_no_prefix[: -len(".weight")]
        else:
            module_path = "mtp." + key_no_prefix

        skip_endings = (
            "hc_norm", "q_norm", "k_norm", "norm_key", "norm_query", "norm_conv",
            "q_layernorm", "k_layernorm", "norm.weight", "A_log", "dt_bias",
            "pre_fc_norm_hidden", "pre_fc_norm_embedding",
        )
        if any(module_path.endswith(e) for e in skip_endings):
            return False

        if not key_no_prefix.endswith(".weight"):
            return False

        if w.ndim < 2 or w.shape[-1] % GROUP_SIZE != 0:
            return False

        if module_path.endswith("mlp.gate") or module_path.endswith("shared_expert_gate"):
            return {"group_size": GROUP_SIZE, "bits": 8}

        return True

    return mtp_pred


def collect_mtp_from_bf16(bf16_dir: Path) -> dict:
    """Read all mtp.* tensors from the BF16 shards, one shard at a time."""
    shards = sorted(bf16_dir.glob("model-*-of-*.safetensors"))
    mtp_raw: dict = {}
    for sh in shards:
        w = mx.load(str(sh))
        mtp_keys = {k: v for k, v in w.items() if k.startswith("mtp.")}
        if mtp_keys:
            mtp_raw.update(mtp_keys)
        del w
        gc.collect()
    print(f"[collect] {len(mtp_raw)} raw mtp.* tensors", flush=True)
    return mtp_raw


def quantize_mtp_weights_dict(mtp_raw: dict, args: ModelArgs) -> dict:
    """Sanitize and quantize mtp.* tensors."""
    mtp_sanitized = sanitize_mtp_weights(mtp_raw, args)
    pred = build_mtp_quant_predicate(args)

    out = {}
    n_quant4 = n_quant8 = n_passthrough = 0

    for k, v in mtp_sanitized.items():
        full_key = f"mtp.{k}"
        r = pred(k, v)
        if r is False or r is None:
            out[full_key] = v
            n_passthrough += 1
        else:
            bits = r.get("bits", BITS) if isinstance(r, dict) else BITS
            gs = r.get("group_size", GROUP_SIZE) if isinstance(r, dict) else GROUP_SIZE
            wq, scales, biases = mx.quantize(v, group_size=gs, bits=bits)
            module_path = full_key[: -len(".weight")] if full_key.endswith(".weight") else full_key
            out[full_key] = wq
            out[f"{module_path}.scales"] = scales
            out[f"{module_path}.biases"] = biases
            if bits == 4:
                n_quant4 += 1
            else:
                n_quant8 += 1

    print(
        f"[quant] {n_quant4} @4bit, {n_quant8} @8bit, {n_passthrough} pass-through",
        flush=True,
    )
    return out


def write_mtp_shard(mtp_quantized: dict, dst: Path, dry_run: bool) -> Path:
    """Write mtp-weights.safetensors into the 4-bit directory."""
    # mx.eval forces pending MLX graph computation (this is graph evaluation, not code eval)
    mx.eval({"done": mx.array(1)})
    out_path = dst / MTP_SHARD_NAME
    if not dry_run:
        mx.save_safetensors(str(out_path), mtp_quantized)
        os.sync()
        stat = out_path.stat()
        print(
            f"[write] {out_path.name} -> {stat.st_size / 1e6:.1f} MB  ({len(mtp_quantized)} tensors)",
            flush=True,
        )
    else:
        print(
            f"[dry-run] would write {out_path.name} with {len(mtp_quantized)} tensors",
            flush=True,
        )
    return out_path


def patch_index(mtp_quantized: dict, dst: Path, dry_run: bool) -> None:
    """Add mtp.* tensors to model.safetensors.index.json."""
    idx_path = dst / "model.safetensors.index.json"
    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]

    added = 0
    for k in mtp_quantized:
        if k not in weight_map:
            weight_map[k] = MTP_SHARD_NAME
            added += 1

    total_size = sum(
        (dst / sh).stat().st_size
        for sh in set(weight_map.values())
        if (dst / sh).exists()
    )
    idx["metadata"]["total_size"] = total_size

    if not dry_run:
        json.dump(idx, open(idx_path, "w"), indent=2)
        print(f"[index] patched {added} mtp keys into {idx_path.name}", flush=True)
    else:
        print(f"[dry-run] would patch {added} mtp keys into index.json", flush=True)


def patch_config(dst: Path, mtp_quantized: dict, dry_run: bool) -> None:
    """Patch the 4-bit config.json to add load_mtp=true."""
    cfg_path = dst / "config.json"
    cfg = json.load(open(cfg_path))

    tc = cfg.get("text_config", {})
    tc["load_mtp"] = True
    cfg["text_config"] = tc

    quant = cfg.get("quantization", {})
    for k in mtp_quantized:
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        module_path = k[: -len(".weight")] if k.endswith(".weight") else k
        if "shared_expert_gate" in module_path or (
            module_path.endswith(".gate") and "mlp.gate" in module_path
        ):
            quant[module_path] = {"group_size": GROUP_SIZE, "bits": 8, "mode": "affine"}

    cfg["quantization"] = quant
    cfg["quantization_config"] = quant

    if not dry_run:
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print("[config] patched config.json with load_mtp=true", flush=True)
    else:
        print("[dry-run] would patch config.json with load_mtp=true", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Quantize and inject mtp.* tensors into an existing 4-bit checkpoint"
    )
    ap.add_argument("--bf16-dir", required=True, type=Path,
                    help="Path to the original BF16 HF checkpoint containing mtp.* tensors")
    ap.add_argument("--dst-dir", required=True, type=Path,
                    help="Path to the existing 4-bit MLX checkpoint to patch")
    ap.add_argument("--dry-run", action="store_true")
    cli = ap.parse_args(argv)

    bf16_dir: Path = cli.bf16_dir
    dst: Path = cli.dst_dir
    dry_run = cli.dry_run

    if not bf16_dir.is_dir():
        print(f"ERROR: --bf16-dir {bf16_dir} not found", file=sys.stderr)
        return 2
    if not dst.is_dir():
        print(f"ERROR: --dst-dir {dst} not found", file=sys.stderr)
        return 2

    t0 = time.time()
    stage("start")

    free = vm_free_gb()
    if free < 50.0:
        raise MemoryError(f"Only {free:.1f} GB free; need >= 50 GB. Abort.")

    cfg = json.load(open(dst / "config.json"))
    args = ModelArgs.from_dict(cfg)
    print(f"[cfg] hidden={args.hidden_size} experts={args.num_experts}", flush=True)

    stage("collecting mtp tensors from BF16 shards")
    mtp_raw = collect_mtp_from_bf16(bf16_dir)
    stage("after collect")

    stage("sanitize + quantize")
    mtp_quantized = quantize_mtp_weights_dict(mtp_raw, args)
    del mtp_raw
    gc.collect()
    stage("after quantize")

    out_path = write_mtp_shard(mtp_quantized, dst, dry_run)

    if not dry_run and out_path.exists():
        patch_index(mtp_quantized, dst, dry_run)
        patch_config(dst, mtp_quantized, dry_run)
        stat = out_path.stat()
        print(f"[verify] {out_path} size={stat.st_size} bytes", flush=True)
    elif dry_run:
        patch_index(mtp_quantized, dst, dry_run)
        patch_config(dst, mtp_quantized, dry_run)

    print(f"[done] wall time {time.time()-t0:.1f}s  RSS={rss_gb():.1f}GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
