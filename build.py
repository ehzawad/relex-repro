"""Build HF checkpoints theta_0 + sum_s w_s Delta_s for many predictors in one streaming pass over the delta store.

  python relex/build.py --grams <grams.pt> --spec <spec.json> --outdir <dir> [--round-bf16]

spec.json: {"<name>": {"method": "relex", "obs": [..], "target": T, "rank": 1, "fit": "linear"}
            | {"method": "recon", "fit_steps": [..], "target": s, "rank": r}
            | {"method": "rawls", "obs": [..], "target": T} | {"method": "twopoint", "t0": a, "t1": b, "target": T}
            | {"method": "expo", "t1": b, "alpha": x} | {"method": "rlvr", "step": s} | {"method": "base"}}
A checkpoint that already has model.safetensors is skipped. Also writes <outdir>/<name>/relex_info.json with the
weights' diagnostics (mean per-tensor cosine / norm ratio vs every stored step >= the last observed one).
"""
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relex import (Rows, align, expo_w, onehot_w, rawls_w, recon_w, relex_w, twopoint_w)  # noqa: E402
from store import Store, load_state  # noqa: E402


def weights(spec, G, steps):
    m = spec["method"]
    if m == "relex":
        return relex_w(G, steps, spec["obs"], spec["target"], spec.get("rank", 1), spec.get("fit", "linear"))
    if m == "recon":
        return recon_w(G, steps, spec["fit_steps"], spec["target"], spec.get("rank", 1))
    if m == "rawls":
        return rawls_w(steps, spec["obs"], spec["target"])
    if m == "twopoint":
        return twopoint_w(steps, spec["t0"], spec["t1"], spec["target"])
    if m == "expo":
        return expo_w(steps, spec["t1"], spec["alpha"])
    if m == "relex_normmatch":
        # RELEX direction, rescaled per tensor to the norm of a real stored step (council control: magnitude vs direction)
        w = relex_w(G, steps, spec["obs"], spec["target"], spec.get("rank", 1), spec.get("fit", "linear"))
        j = steps.index(spec["match_step"]); ww = float(w @ G @ w)
        return w * ((float(G[j, j]) / ww) ** 0.5 if ww > 0 else 0.0)
    if m == "rlvr":
        return onehot_w(steps, spec["step"])
    if m == "base":
        return torch.zeros(len(steps), dtype=torch.float64)
    raise ValueError(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grams", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--round-bf16", action="store_true", help="use bf16-rounded checkpoint deltas (must match the grams)")
    ap.add_argument("--only", action="append", help="build only this spec name (repeatable): bounds host memory to one output")
    a = ap.parse_args()
    gd = torch.load(a.grams); steps, grams = gd["steps"], gd["grams"]
    assert bool(gd.get("round_bf16")) == a.round_bf16, "grams and --round-bf16 disagree"
    st = Store(gd["store"]); spec = json.loads(Path(a.spec).read_text()); outdir = Path(a.outdir)
    if a.only:
        missing = set(a.only) - set(spec)
        if missing:
            ap.error(f"unknown spec names: {sorted(missing)}")
        spec = {k: spec[k] for k in a.only}
    todo = {k: v for k, v in spec.items() if not (outdir / k / "model.safetensors").exists()}
    print(f"{len(todo)} of {len(spec)} checkpoints to build", flush=True)
    if not todo:
        return
    bdir = snapshot_download(st.meta["base"], allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    base = load_state(bdir, torch.bfloat16); rows = Rows(st, base if a.round_bf16 else None)
    W = {k: {n: weights(v, grams[n], steps) for n in st.names} for k, v in todo.items()}
    out = {k: {} for k in todo}; t0 = time.time()
    from relex import chunks
    for i, n in enumerate(st.names):
        need = sorted({s for k in todo for s, x in zip(steps, W[k][n].tolist()) if x != 0})
        shape = st.shape(n)
        for k in todo:
            out[k][n] = torch.empty(shape, dtype=torch.bfloat16)
        # row chunks small enough for one fp32 accumulator per output plus one delta row
        for r0, r1 in chunks(shape, len(todo) + 1, budget=8 * 2 ** 30):
            acc = {k: base[n][r0:r1].float().clone() for k in todo}
            for s in need:
                x = rows.get(n, s, r0, r1) if len(shape) > 0 and shape[0] > 0 else rows.get(n, s)
                for k in todo:
                    c = float(W[k][n][steps.index(s)])
                    if c != 0:
                        acc[k] += c * x
            for k in todo:
                out[k][n][r0:r1] = acc[k].to(torch.bfloat16)
        if i % 60 == 0:
            print(f"{i}/{len(st.names)} {time.time() - t0:.0f}s", flush=True)
    for k in todo:
        d = outdir / k; d.mkdir(parents=True, exist_ok=True)
        sd = out[k]
        if "lm_head.weight" in base and "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
        for n in base:                       # untracked base tensors (e.g. Qwen3.5's mtp.* head) are copied unchanged
            if n not in sd:
                sd[n] = base[n].clone()
        save_file({n: v.contiguous() for n, v in sd.items()}, str(d / "model.safetensors"), metadata={"format": "pt"})
        for f in Path(bdir).iterdir():
            if f.suffix in (".json", ".txt", ".jinja") and f.name != "model.safetensors.index.json":
                shutil.copy(f, d / f.name)
        obs_last = max(todo[k].get("obs", [0]) or [0]) if todo[k]["method"] in ("relex", "rawls") else 0
        diag = {}
        for s in steps:
            if s >= obs_last:
                cs = [align(grams[n], steps, W[k][n], s) for n in st.names]
                cs = [c for c in cs if not np.isnan(c[0])]
                diag[s] = {"cos": float(np.mean([c[0] for c in cs])), "norm_ratio": float(np.median([c[1] for c in cs]))}
        (d / "relex_info.json").write_text(json.dumps({"spec": todo[k], "round_bf16": a.round_bf16, "align": diag}, indent=1))
        out[k] = None
        print("wrote", d, flush=True)


if __name__ == "__main__":
    main()
