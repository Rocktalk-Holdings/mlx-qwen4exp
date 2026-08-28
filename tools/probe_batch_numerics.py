#!/usr/bin/env python3.11
"""Decisive probe: are T=1 and T=2 forward passes bit-identical on this mlx?

Loads the 4-bit model once, then compares:
  A) logits of model([[a]]) at position 0  vs  model([[a, b]]) at position 0
     (fresh caches both times, same prompt prefill first)
  B) the KV cache K-block written for position 0 under each path

If max|diff| > 0 the batch-verify bonus-token trick can NEVER be bit-equal to
single-token greedy on this mlx version, independent of snapshot correctness.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import mlx.core as mx
import numpy as np

from tools.run_mlx import load_model

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = load_model(Path(MODEL_DIR))
print(f"[probe] mlx {mx.__version__}")

prompt = tokenizer.encode("In the beginning God created the heavens and the")
a, b = 279, 1879  # two arbitrary continuation tokens

def fresh_run(tokens_after_prefill):
    model.reset_ple_state()
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    mx.eval(logits)
    out = model(mx.array([tokens_after_prefill]), cache=cache)
    mx.eval(out)
    # grab each QSA layer's cached K for the first new position
    from mlx_qwen4exp.attention import QSACache
    kblocks = []
    for c in cache:
        if isinstance(c, QSACache):
            pos = len(prompt)
            kblocks.append(np.array(c.kv.keys[..., pos, :].astype(mx.float32)))
    return np.array(out[:, 0, :].astype(mx.float32)), kblocks

logits_t1, k_t1 = fresh_run([a])
logits_t2, k_t2 = fresh_run([a, b])

ld = np.abs(logits_t1 - logits_t2).max()
print(f"[probe] logits pos0 max|T1-T2| = {ld:.3e}")
kd = max(np.abs(x - y).max() for x, y in zip(k_t1, k_t2))
print(f"[probe] cached-K pos0 max|T1-T2| = {kd:.3e}")

am1, am2 = int(np.argmax(logits_t1)), int(np.argmax(logits_t2))
print(f"[probe] argmax T1={am1} T2={am2} {'MATCH' if am1 == am2 else 'FLIP'}")
print("[probe] VERDICT:", "BIT-IDENTICAL - batch trick can be lossless" if ld == 0 and kd == 0
      else "NUMERICS DIVERGE - bit-equality to plain greedy impossible on this mlx; claim must be reframed")
