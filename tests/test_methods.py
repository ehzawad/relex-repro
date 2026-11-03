"""Check the Gram-matrix weights against direct computations on a random trajectory (CPU)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from relex import relex_w, recon_w, rawls_w, twopoint_w, rank1_stats, align

rng = np.random.default_rng(0)
d, steps = 5000, list(range(1, 41)) + [60, 80, 100]
v = rng.normal(size=d); v /= np.linalg.norm(v)
M = np.stack([(0.3 * t + 1.0) * v + 0.05 * rng.normal(size=d) for t in steps])          # rank-1 linear + noise
G = torch.tensor(M @ M.T)
obs = list(range(1, 31)); T = 100
# direct RELEX: SVD of observed rows, top right vector, project, linear fit, predict
Mo = M[:30]; U, S, Vt = np.linalg.svd(Mo, full_matrices=False); v1 = Vt[0]; c = Mo @ v1
a, b = np.polyfit(np.array(obs, float), c, 1); direct = (a * T + b) * v1
mine = relex_w(G, steps, obs, T).numpy() @ M
assert np.allclose(mine, direct, atol=1e-8), np.abs(mine - direct).max()
# rank 3
direct3 = sum(np.polyval(np.polyfit(np.array(obs, float), Mo @ Vt[i], 1), T) * Vt[i] for i in range(3))
assert np.allclose(relex_w(G, steps, obs, T, rank=3).numpy() @ M, direct3, atol=1e-8)
# reconstruction of a stored step outside the window
j = steps.index(80); assert np.allclose(recon_w(G, steps, obs, 80).numpy() @ M, (M[j] @ v1) * v1, atol=1e-8)
# raw per-element least squares
co = np.polyfit(np.array(obs, float), Mo, 1); assert np.allclose(rawls_w(steps, obs, T).numpy() @ M, co[0] * T + co[1], atol=1e-8)
# two-point
assert np.allclose(twopoint_w(steps, 10, 30, T).numpy() @ M, M[9] + (T - 10) / 20 * (M[29] - M[9]))
assert np.allclose(twopoint_w(steps, 0, 30, T).numpy() @ M, T / 30 * M[29])
# alignment from G equals direct cosine
w = relex_w(G, steps, obs, T); p = w.numpy() @ M; k = steps.index(100)
cs, nr = align(G, steps, w, 100)
assert abs(cs - p @ M[k] / np.linalg.norm(p) / np.linalg.norm(M[k])) < 1e-10 and abs(nr - np.linalg.norm(p) / np.linalg.norm(M[k])) < 1e-10
st = rank1_stats(G, steps, obs); assert st["ev1"] > 0.5 and st["r2"] > 0.99, st
print("all method checks passed", st, "cos to step 100", round(cs, 4), "norm ratio", round(nr, 4))
