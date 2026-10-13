"""Controlled synthetic representation: x = T*u + N*v + e, with
  overlap c = cos(u, v) (geometric entanglement in the representation), and
  training correlation rho = P(N=T) (shortcut opportunity in the probe's training data),
varied independently. A nuisance 'edit' is exact: x' = x + (1-2N) v. Uses the same extractor code and CF
definition as the real experiments. Noise is anisotropic (power-law spectrum) like real residual streams.
-> results/rev_synthetic.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dv import validity as V  # noqa: E402
from dv.extractors import fit_extractor  # noqa: E402

D, NTR, NTE = 256, 2000, 2000
rng = np.random.RandomState(0)
spec = 1.0 / np.arange(1, D + 1) ** 0.7               # anisotropic noise spectrum
Q, _ = np.linalg.qr(rng.randn(D, D))
L = Q * np.sqrt(spec / spec.mean())                    # noise covariance factor


def make(n, rho, u, v, rs):
    t = rs.randint(0, 2, n)
    nn = np.where(rs.rand(n) < rho, t, 1 - t)
    x = np.outer(t, u) + np.outer(nn, v) + rs.randn(n, D) @ L.T * 1.2
    return x, t, nn


rows = []
for c in [0.0, 0.3, 0.6, 0.9]:
    for rho in [0.5, 0.7, 0.9]:
        for rep in range(5):
            rs = np.random.RandomState(100 * rep + 7)
            u = rs.randn(D)
            u /= np.linalg.norm(u)
            w = rs.randn(D)
            w -= (w @ u) * u
            w /= np.linalg.norm(w)
            v = c * u + np.sqrt(1 - c ** 2) * w         # |v| = 1, cos(u, v) = c
            xtr, ttr, ntr = make(NTR, rho, u, v, rs)
            xte, tte, nte = make(NTE, 0.5, u, v, rs)    # decorrelated evaluation pool
            mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
            Z = (xtr - mu) / sd
            for name, params in [("dom", {}), ("lr_C0.01", {"C": 0.01}), ("lr_C1", {"C": 1.0}),
                                 ("pls_k8", {"k": 8}), ("ne_lr_C0.01", {"C": 0.01})]:
                wv, b = fit_extractor(name, params, Z, ttr, ntr)
                f = lambda x: ((x - mu) / sd) @ wv + b  # noqa: E731
                s, s_cf = f(xte), f(xte + np.outer(1 - 2 * nte, v))
                dT = V.delta_t(s, tte, nte)
                sign = np.where(nte == 0, 1.0, -1.0)    # flipping N=0 -> 1 is the T-congruent direction
                xi, ti, ni = make(NTE, rho, u, v, rs)   # in-distribution test for decodability
                rows.append(dict(overlap=c, rho=rho, rep=rep, extractor=name,
                                 dec=V.dec_auroc(f(xi), ti, ni, 0.5) if rho == 0.5 else
                                 __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(ti, f(xi)),
                                 cf=V.cf_effect(s, s_cf, sign, dT), conflict=V.conflict_auroc(s, tte, nte)))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results/rev_synthetic.csv", index=False)
print(R.groupby(["overlap", "rho", "extractor"])[["dec", "cf", "conflict"]].mean().round(3).unstack("extractor")["cf"].to_string())
print(R.groupby(["overlap", "rho", "extractor"])[["dec"]].mean().round(3).unstack("extractor").to_string())
