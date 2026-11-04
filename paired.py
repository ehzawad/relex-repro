"""Paired differences on MATH between checkpoints (same 5000 problems): diff in points with a 95% bootstrap CI.
  python relex/paired.py --evals <dir> a:b [a:b ...]"""
import argparse, json
from pathlib import Path
import numpy as np


def load(d, n, bench="math"):
    ok = {}
    for l in open(Path(d) / n / f"{bench}.jsonl"):
        if l.strip():
            r = json.loads(l); ok[r["i"]] = float(r["ok"])
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--evals", required=True); ap.add_argument("--bench", default="math"); ap.add_argument("pairs", nargs="+")
    a = ap.parse_args(); rng = np.random.default_rng(0)
    for p in a.pairs:
        x, y = p.split(":"); A, B = load(a.evals, x, a.bench), load(a.evals, y, a.bench); ids = sorted(set(A) & set(B))
        d = np.array([A[i] - B[i] for i in ids]) * 100
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print(f"{x} - {y}: {d.mean():+.2f}  95% CI [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]  (n={len(d)})")


if __name__ == "__main__":
    main()
