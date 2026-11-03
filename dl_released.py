"""Download the authors' released RLVR trajectory (one HF branch per step) and turn each step into a bf16 delta file.

  python relex/dl_released.py --repo relex-rlvr/RLVR-Qwen2.5-Math-1.5B --base Qwen/Qwen2.5-Math-1.5B \
      --steps 1-125,150-500:25 --out $SCRATCH/.../relex/released/q15/deltas

The released checkpoints are bf16, so theta_t - theta_0 is (by Sterbenz) exactly representable in bf16 whenever theta_t and
theta_0 are within a factor of two; we store bf16 and count elements where the round trip is not exact. The raw checkpoint
is deleted right after conversion. Also reports, per step, the fraction of elements whose bf16 delta is exactly zero.
"""
import argparse, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).parent))
from store import Store, load_state, tracked


def parse_steps(s):
    out = []
    for part in s.split(","):
        rng, _, every = part.partition(":")
        a, _, b = rng.partition("-")
        out += list(range(int(a), int(b or a) + 1, int(every or 1)))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--steps", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    out, tmp = Path(a.out), Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    tied = bool(getattr(AutoConfig.from_pretrained(a.base), "tie_word_embeddings", False))
    base_dir = snapshot_download(a.base, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    base = load_state(base_dir, torch.bfloat16)
    names = tracked(base, tied)
    if not (out / "meta.json").exists():
        Store.create(out, a.base, "bfloat16", tied, {k: list(base[k].shape) for k in names})
    st = Store(out); todo = [s for s in parse_steps(a.steps) if s not in st.steps]
    print(f"{len(todo)} steps to fetch ({len(st.steps)} present); tied={tied}; {len(names)} tensors", flush=True)
    stats_f = out / "convert_stats.jsonl"

    def one(step):
        t0 = time.time(); d = tmp / f"step_{step}"
        snapshot_download(a.repo, revision=f"step_{step}", local_dir=str(d), allow_patterns=["*.safetensors", "config.json"])
        sd = load_state(d, torch.bfloat16)
        deltas, inexact, zero, n = {}, 0, 0, 0
        for k in names:
            exact = sd[k].float() - base[k].float()
            q = exact.to(torch.bfloat16)
            inexact += int((q.float() != exact).sum()); zero += int((q == 0).sum()); n += q.numel()
            deltas[k] = q
        st.write(step, deltas); shutil.rmtree(d, ignore_errors=True)
        rec = {"step": step, "inexact_frac": inexact / n, "zero_frac": zero / n, "sec": round(time.time() - t0, 1)}
        with open(stats_f, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(one, todo):
            pass
    print("done", flush=True)


if __name__ == "__main__":
    main()
