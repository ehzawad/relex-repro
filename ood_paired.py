"""Paired OOD differences (macro mean over benchmarks of avg@k), problem-clustered bootstrap stratified by benchmark.
  python relex/ood_paired.py --evals <dir> a:b [a:b ...]"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load(d, n):
    acc = defaultdict(list)
    for l in open(Path(d) / n / "ood.jsonl"):
        if l.strip():
            r = json.loads(l); acc[(r["ds"], r["i"])].append(float(r["ok"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--evals", required=True); ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("pairs", nargs="+"); a = ap.parse_args(); rng = np.random.default_rng(0)
    for p in a.pairs:
        x, y = p.split(":"); A, Bm = load(a.evals, x), load(a.evals, y)
        by = defaultdict(list)
        for k in sorted(set(A) & set(Bm)):
            by[k[0]].append(A[k] - Bm[k])
        ds = sorted(by); d = [np.array(by[s]) for s in ds]
        point = 100 * np.mean([v.mean() for v in d])
        bs = 100 * np.mean([v[rng.integers(0, len(v), (a.B, len(v)))].mean(1) for v in d], axis=0)
        print(f"{x} - {y}: {point:+.2f}  95% CI [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]  "
              f"(benchmarks={len(ds)}, problems={sum(len(v) for v in d)})")


if __name__ == "__main__":
    main()
