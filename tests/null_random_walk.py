"""Null model for the rank-1 statistics: per 'tensor', Delta_t = drift * t * v + sum_{s<=t} noise_s (random walk), d=20000.
Reports median rank-1 explained variance and R^2 of the c_1 line over T=75, for several drift-to-noise ratios."""
import numpy as np
rng = np.random.default_rng(0); d, T = 20000, 75
t = np.arange(1, T + 1)
for snr in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
    ev, r2 = [], []
    for _ in range(40):
        v = rng.normal(size=d); v /= np.linalg.norm(v)
        M = np.cumsum(rng.normal(size=(T, d)) / np.sqrt(d), axis=0) + snr * t[:, None] * v[None, :]
        U, S, Vt = np.linalg.svd(M, full_matrices=False); c = U[:, 0] * S[0]
        ev.append(S[0] ** 2 / (S ** 2).sum()); p = np.polyfit(t, c, 1)
        r2.append(1 - ((c - np.polyval(p, t)) ** 2).sum() / ((c - c.mean()) ** 2).sum())
    print(f"drift/step-noise {snr:4}: rank-1 EV median {np.median(ev):.3f}  R^2 median {np.median(r2):.3f}  frac R^2>0.98 {np.mean(np.array(r2) > 0.98):.2f}")
