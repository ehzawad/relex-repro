"""Weight-level comparison of checkpoints against a reference, as deltas from a common base.

  python relex/compare_ckpt.py --base Qwen/Qwen2.5-Math-1.5B --ref relex-rlvr/RELEX-Qwen2.5-Math-1.5B --cands dir1,dir2
Per candidate: per-tensor cosine of (cand - base) vs (ref - base), whole-model relative L2 ||cand - ref|| / ||ref - base||,
and norm ratio ||cand - base|| / ||ref - base||. lm_head is skipped when tied.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import load_state  # noqa: E402


def get(p):
    return load_state(p if Path(p).exists() else snapshot_download(p, allow_patterns=["*.safetensors"]), torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--cands", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    base, ref = get(a.base), get(a.ref)
    names = [n for n in base if n != "lm_head.weight"]
    res = {}
    for c in a.cands.split(","):
        cd = get(c); cos, num, den, nc = [], 0.0, 0.0, 0.0
        for n in names:
            dr, dc = ref[n] - base[n], cd[n] - base[n]
            a2, b2 = float((dr * dr).sum()), float((dc * dc).sum())
            if a2 > 0 and b2 > 0:
                cos.append(float((dr * dc).sum()) / (a2 * b2) ** 0.5)
            num += float(((cd[n] - ref[n]) ** 2).sum()); den += a2; nc += b2
        cos = np.array(cos)
        res[c] = {"cos_median": float(np.median(cos)), "cos_p5": float(np.percentile(cos, 5)),
                  "frac_cos_gt_0.99": float((cos > 0.99).mean()), "rel_l2": (num / den) ** 0.5, "norm_ratio": (nc / den) ** 0.5}
        print(c, json.dumps(res[c]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
