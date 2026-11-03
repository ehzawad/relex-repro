"""Weight-space diagnostics computed from Gram matrices alone (no checkpoints, no GPU).

  python relex/analyze_grams.py --grams g1.pt[,g2.pt] --out stats.json

Per Gram file:
  windows   for prefixes 1..T (T in --windows) and for all stored steps: per-tensor rank-1 explained variance (sigma_1^2 /
            sum sigma^2) and R^2 of the line fitted to c_1(t), summarised by median / p10 / fraction with R^2 > 0.98,
            overall and per module type.
  align     the paper's Appendix B: mean per-tensor cosine and median norm ratio between the predicted delta and the stored
            delta at every stored step >= T_obs, for reconstruction (fit on all stored steps) and RELEX extrapolation from
            each window; plus raw per-element least squares for comparison.
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relex import align, rank1_stats, rawls_w, recon_w, relex_w  # noqa: E402


def mtype(n):
    m = re.search(r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|input_layernorm|post_attention_layernorm|embed_tokens)", n)
    return m.group(1) if m else ("norm" if n.endswith("norm.weight") else "other")


def summ(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    if not len(v):
        return {}
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "mean": float(v.mean()), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grams", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="25,50,75,100,125"); ap.add_argument("--matrices-only", action="store_true")
    a = ap.parse_args()
    res = {}
    for gp in a.grams.split(","):
        gd = torch.load(gp); steps, grams = gd["steps"], gd["grams"]
        names = [n for n in grams if (not a.matrices_only) or n.endswith("proj.weight")]
        names = [n for n in names if float(grams[n].diagonal().sum()) > 0]          # tensors that moved at all
        wins = [int(x) for x in a.windows.split(",") if int(x) <= max(steps)]
        out = {"steps": steps, "n_tensors": len(names), "windows": {}, "align": {}}
        uni = [s for s in steps if s % 25 == 0]                               # uniform subsample of the whole run
        for T in wins + ["all", "u25"]:
            obs = uni if T == "u25" else [s for s in steps if T == "all" or s <= T]
            st = {n: rank1_stats(grams[n], steps, obs) for n in names}
            r2 = [s["r2"] for s in st.values()]; ev = [s["ev1"] for s in st.values()]
            byt = {}
            for n, s in st.items():
                byt.setdefault(mtype(n), []).append(s)
            out["windows"][str(T)] = {
                "ev1": summ(ev), "r2": summ(r2), "frac_r2_gt_0.98": float(np.mean([x > 0.98 for x in r2 if not np.isnan(x)])),
                "by_type": {k: {"ev1_median": float(np.nanmedian([s["ev1"] for s in v])), "r2_median": float(np.nanmedian([s["r2"] for s in v]))}
                            for k, v in sorted(byt.items())}}
        later = [s for s in steps if s >= 100 and (s % 25 == 0)]

        def curve(wf):
            c = {}
            for s in later:
                pairs = [align(grams[n], steps, wf(n, s), s) for n in names]
                pairs = [p for p in pairs if not np.isnan(p[0])]
                c[s] = {"cos": float(np.mean([p[0] for p in pairs])), "norm_ratio": float(np.median([p[1] for p in pairs]))}
            return c

        out["align"]["recon_u25"] = curve(lambda n, s: recon_w(grams[n], steps, uni, s))
        for T in wins:
            obs = [s for s in steps if s <= T]
            out["align"][f"relex_c{T}"] = curve(lambda n, s, obs=obs: relex_w(grams[n], steps, obs, s))
            out["align"][f"rawls_c{T}"] = curve(lambda n, s, obs=obs: rawls_w(steps, obs, s))
        res[gp] = out
        w = out["windows"]
        print(gp, {k: (round(v["ev1"].get("median", float("nan")), 3), round(v["r2"].get("median", float("nan")), 3),
                       round(v["frac_r2_gt_0.98"], 3)) for k, v in w.items()}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
