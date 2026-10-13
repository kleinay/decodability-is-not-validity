"""Concept extractors. Every extractor is a *linear measurement* s(x) = w . z(x) + b on
standardized activations z, oriented so that larger s means "more target". Treating all
of them as scalar instruments (rather than as subspaces) is what lets the same validity
battery apply to a rank-1 probe and a rank-16 PLS subspace alike.

Families:
  dom        difference of class means (CAV-style)
  lr_C       L2 logistic regression, C in a grid
  pls_k      partial least squares, k components, score = PLS regression prediction
  ne_lr_C    nuisance-erased probe: project out the within-target-class nuisance mean
             differences (so N is linearly guarded conditional on T in the training data,
             cf. LEACE), then L2 logistic regression
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
PLS_GRID = [1, 2, 4, 8, 16]


def configs():
    out = [("dom", {})]
    out += [(f"lr_C{c:g}", {"C": c}) for c in C_GRID]
    out += [(f"pls_k{k}", {"k": k}) for k in PLS_GRID]
    out += [(f"ne_lr_C{c:g}", {"C": c}) for c in C_GRID]
    return out


def family(name: str) -> str:
    return name.split("_")[0] if not name.startswith("ne_") else "ne_lr"


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def nuisance_eraser(Z, t, n):
    """Orthonormal basis of the within-T-class nuisance mean differences."""
    dirs = []
    for c in (0, 1):
        m = t == c
        if (n[m] == 1).sum() > 1 and (n[m] == 0).sum() > 1:
            dirs.append(Z[m & (n == 1)].mean(0) - Z[m & (n == 0)].mean(0))
    Q, _ = np.linalg.qr(np.stack(dirs, 1))
    return Q


def fit_extractor(name, params, Z, t, n):
    """Returns (w, b): s(z) = z @ w + b. Z is already standardized on the training set."""
    if name == "dom":
        w = _unit(Z[t == 1].mean(0) - Z[t == 0].mean(0))
        return w, 0.0
    if name.startswith("lr_"):
        clf = LogisticRegression(C=params["C"], max_iter=5000, tol=1e-4).fit(Z, t)
        return clf.coef_.ravel(), float(clf.intercept_[0])
    if name.startswith("pls_"):
        pls = PLSRegression(n_components=params["k"], scale=False).fit(Z, t.astype(float))
        coef = pls.coef_.reshape(-1) if pls.coef_.shape[0] == 1 else pls.coef_.ravel()
        return coef, 0.0
    if name.startswith("ne_lr_"):
        Q = nuisance_eraser(Z, t, n)
        Ze = Z - Z @ Q @ Q.T
        clf = LogisticRegression(C=params["C"], max_iter=5000, tol=1e-4).fit(Ze, t)
        w = clf.coef_.ravel()
        w = w - Q @ (Q.T @ w)  # s(z) = w . P z  ==  (P w) . z  since P is symmetric
        return w, float(clf.intercept_[0])
    raise ValueError(name)
