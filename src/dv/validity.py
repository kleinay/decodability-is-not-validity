"""Validity battery for a scalar concept measurement s(x).

All observational metrics are computed on an evaluation pool with labels (T, N) by
*reweighting* its four (T, N) cells to a chosen agreement rate
rho = P(N=1 | T=1) = P(N=0 | T=0) with balanced T -- so one pool yields the in-distribution
test for any training regime as well as decorrelated / reversed variants.

  dec        AUROC(s -> T) at the training regime's rho        (decodability, what is usually reported)
  conflict   AUROC(s -> T) on the T!=N cells only (rho -> 0)     (criterion validity under correlation reversal)
  leak       mean_t AUROC(s -> N | T=t) - 0.5                    (discriminant validity: nuisance signal holding target fixed)
  cf_n       mean over nuisance-counterfactual pairs of sign * (s_b - s_a) / Delta_T
             where Delta_T = E[s|T=1] - E[s|T=0] on the decorrelated pool
             (fraction of a full target-class shift caused by flipping only the nuisance)
  cf_t       same for target-counterfactual pairs (CEBaB only; ideal ~ 1)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def cell_weights(t, n, rho):
    w = np.zeros(len(t))
    target = {(1, 1): rho, (1, 0): 1 - rho, (0, 0): rho, (0, 1): 1 - rho}
    for (ct, cn), p in target.items():
        m = (t == ct) & (n == cn)
        if m.sum():
            w[m] = 0.5 * p / m.sum()
    return w


def dec_auroc(s, t, n, rho):
    w = cell_weights(t, n, rho)
    keep = w > 0
    return roc_auc_score(t[keep], s[keep], sample_weight=w[keep])


def conflict_auroc(s, t, n):
    m = t != n
    return roc_auc_score(t[m], s[m])


def leak(s, t, n):
    """Signed: positive when the score treats the nuisance value that co-occurs with T=1 as evidence for T.
    Opposite-direction dependence in the two target classes cancels here; see leak_parts / leak_abs."""
    vals = [roc_auc_score(n[t == c], s[t == c]) for c in (0, 1)]
    return float(np.mean(vals)) - 0.5


def leak_parts(s, t, n):
    """(AUROC(s->N | T=0), AUROC(s->N | T=1), mean_t |AUROC_t - 0.5|)."""
    a0, a1 = (roc_auc_score(n[t == c], s[t == c]) for c in (0, 1))
    return a0, a1, 0.5 * (abs(a0 - 0.5) + abs(a1 - 0.5))


def delta_t(s, t, n):
    w = cell_weights(t, n, 0.5)
    return float(np.sum(w * s * (t == 1)) / np.sum(w * (t == 1)) - np.sum(w * s * (t == 0)) / np.sum(w * (t == 0)))


def cf_effect(s_a, s_b, sign, dT):
    return float(np.mean(sign * (s_b - s_a)) / dT)
