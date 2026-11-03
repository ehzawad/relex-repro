"""RELEX and its baselines, expressed through per-tensor Gram matrices of the delta trajectory.

Every predictor here outputs Delta_hat = sum_s w_s Delta_s over stored steps s, so it is fully described by a weight vector
w (per tensor for the SVD methods, shared for the raw-weight ones):

  RELEX rank r, fit f   G_obs = U diag(S^2) U^T; C = U_r S_r (coefficients, = M V_r); fit each column of C against step;
                        c_hat = f(T); Delta_hat = sum_i c_hat_i v_i, v_i = M^T u_i / S_i  ->  w = sum_i c_hat_i u_i / S_i
  reconstruction        c_i(s) = <Delta_s, v_i> = G[s,obs] u_i / S_i (Algorithm 1 when s is in the fit window)
  raw per-element LS    fit a line to every weight separately over the observed steps (the paper's "Raw" ablation)
  two-point             Delta_t0 + (T - t0)/(t1 - t0) (Delta_t1 - Delta_t0)        (Weight Extrap., Wang et al.)
  ExPO                  (1 + alpha) Delta_t1

Diagnostics from G alone: rank-1 explained variance, R^2 of the c_1 line, and for any w the cosine and norm ratio
against a stored step s (cos = w^T G[:,s] / sqrt(w^T G w G[s,s])), i.e. the paper's Appendix B analysis.

  python relex/relex.py gram --store <deltas> --out <grams.pt> [--round-bf16] [--steps 1-125,150-500:25]
"""
import argparse, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import Store, load_state  # noqa: E402


# ------------------------------------------------------------------------------------------------------------ loading
class Rows:
    """Delta rows of a store, optionally as the bf16-rounded checkpoint would give them: bf16(theta_0 + Delta) - theta_0."""

    def __init__(self, store, base=None):
        self.st, self.base = store, base

    def get(self, name, step, r0=None, r1=None):
        x = self.st.rows(name, step, r0, r1)
        if self.base is not None:
            b = self.base[name] if r0 is None else self.base[name][r0:r1]
            b = b.float()
            x = (b + x).to(torch.bfloat16).float() - b
        return x


def chunks(shape, T, budget=2 ** 31):
    """Row ranges so that T x rows x cols float32 stays under `budget` bytes."""
    n = shape[0]; per = (math.prod(shape[1:]) if len(shape) > 1 else 1) * T * 4
    k = max(1, budget // max(per, 1))
    return [(i, min(n, i + k)) for i in range(0, n, k)]


def gram_tensor(rows, name, steps, shape):
    G = torch.zeros(len(steps), len(steps), dtype=torch.float64)
    for r0, r1 in chunks(shape, len(steps)):
        M = torch.stack([rows.get(name, s, r0, r1).reshape(-1) for s in steps])
        G += (M @ M.T).double()
    return G


def cmd_gram(a):
    st = Store(a.store)
    steps = st.steps if not a.steps else [s for s in parse_steps(a.steps) if s in set(st.steps)]
    base = None
    if a.round_bf16:
        from huggingface_hub import snapshot_download
        base = load_state(snapshot_download(st.meta["base"], allow_patterns=["*.safetensors"]), torch.bfloat16)
    rows = Rows(st, base); names = st.names; out = {}
    t0 = time.time()
    torch.set_num_threads(max(1, a.threads // a.workers))

    def one(n):
        return n, gram_tensor(rows, n, steps, st.shape(n))

    with ThreadPoolExecutor(a.workers) as ex:
        for i, (n, G) in enumerate(ex.map(one, names)):
            out[n] = G
            if i % 50 == 0:
                print(f"{i}/{len(names)} {n} {time.time() - t0:.0f}s", flush=True)
    torch.save({"steps": steps, "grams": out, "round_bf16": a.round_bf16, "store": str(a.store)}, a.out)
    print("saved", a.out, f"{time.time() - t0:.0f}s")


def parse_steps(s):
    out = []
    for part in s.split(","):
        rng, _, every = part.partition(":")
        a, _, b = rng.partition("-")
        out += list(range(int(a), int(b or a) + 1, int(every or 1)))
    return sorted(set(out))


# ------------------------------------------------------------------------------------------------------------ methods
def _eig(G):
    lam, U = torch.linalg.eigh(G)
    lam, U = lam.flip(0).clamp(min=0), U.flip(1)
    return lam.sqrt(), U


def _fit(t, y, kind):
    """Least-squares fit of y (T,) against t (T,); returns a callable."""
    deg = {"linear": 1, "poly2": 2, "poly3": 3}[kind]
    tc = t.mean(); sc = max(float(t.std()), 1.0)
    coef = np.polyfit((t - tc) / sc, y, deg)
    return lambda T: float(np.polyval(coef, (T - tc) / sc))


def relex_w(G, steps, obs, target, rank=1, fit="linear"):
    """Weights over `steps` (stored order) of the RELEX prediction at `target` from observed steps `obs`."""
    idx = [steps.index(s) for s in obs]
    S, U = _eig(G[np.ix_(idx, idx)])
    w = torch.zeros(len(steps), dtype=torch.float64)
    t = np.array(obs, dtype=np.float64)
    for i in range(min(rank, len(idx))):
        if S[i] <= S[0] * 1e-6 or S[i] == 0:
            continue
        c = (U[:, i] * S[i]).numpy()                     # c_t = <Delta_t, v_i>
        chat = _fit(t, c, fit)(target)
        w[idx] += chat * U[:, i] / S[i]
    return w


def recon_w(G, steps, fit_steps, target, rank=1):
    """Rank-r projection of the stored Delta_target onto the subspace of `fit_steps` (paper Algorithm 1)."""
    idx = [steps.index(s) for s in fit_steps]; j = steps.index(target)
    S, U = _eig(G[np.ix_(idx, idx)])
    w = torch.zeros(len(steps), dtype=torch.float64)
    for i in range(min(rank, len(idx))):
        if S[i] <= S[0] * 1e-6 or S[i] == 0:
            continue
        ci = float(G[j, idx] @ U[:, i]) / float(S[i])   # <Delta_target, v_i>
        w[idx] += ci * U[:, i] / S[i]
    return w


def rawls_w(steps, obs, target):
    t = np.array(obs, dtype=np.float64); tb = t.mean(); var = ((t - tb) ** 2).sum()
    w = torch.zeros(len(steps), dtype=torch.float64)
    for s, ts in zip(obs, t):
        w[steps.index(s)] = 1 / len(t) + (ts - tb) * (target - tb) / var
    return w


def twopoint_w(steps, t0, t1, target):
    w = torch.zeros(len(steps), dtype=torch.float64); r = (target - t0) / (t1 - t0)
    w[steps.index(t1)] += r
    if t0 != 0:
        w[steps.index(t0)] += 1 - r
    return w


def expo_w(steps, t1, alpha):
    w = torch.zeros(len(steps), dtype=torch.float64); w[steps.index(t1)] = 1 + alpha
    return w


def onehot_w(steps, s):
    w = torch.zeros(len(steps), dtype=torch.float64); w[steps.index(s)] = 1.0
    return w


# -------------------------------------------------------------------------------------------------------- diagnostics
def rank1_stats(G, steps, obs):
    """Rank-1 explained variance and R^2 of the linear fit of c_1 over `obs`."""
    idx = [steps.index(s) for s in obs]
    S, U = _eig(G[np.ix_(idx, idx)])
    tot = float((S ** 2).sum())
    if tot == 0:
        return {"ev1": float("nan"), "r2": float("nan")}
    c = (U[:, 0] * S[0]).numpy(); t = np.array(obs, float)
    p = np.polyfit(t, c, 1); res = c - np.polyval(p, t)
    r2 = 1 - (res ** 2).sum() / max(((c - c.mean()) ** 2).sum(), 1e-300)
    return {"ev1": float(S[0] ** 2) / tot, "r2": float(r2)}


def align(G, steps, w, s):
    j = steps.index(s); ww = float(w @ G @ w); gs = float(G[j, j])
    if ww <= 0 or gs <= 0:
        return float("nan"), float("nan")
    return float(w @ G[:, j]) / math.sqrt(ww * gs), math.sqrt(ww / gs)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gram"); g.add_argument("--store", required=True); g.add_argument("--out", required=True)
    g.add_argument("--steps", default=None); g.add_argument("--round-bf16", action="store_true")
    g.add_argument("--workers", type=int, default=4); g.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    {"gram": cmd_gram}[a.cmd](a)


if __name__ == "__main__":
    main()
