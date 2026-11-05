"""Do the main results depend on how edit pairs are filtered? Recomputes the signed nuisance sensitivity CF (as in
08_cf_variants.py: oriented score change / Delta_T; Civil: mean of both directions) under five pair filters
  all     every generated pair that passed the automatic checks
  j1      accepted by the first judge alone  (judge.parquet:      Qwen3-32B; Amazon: Qwen3.8-27B)
  j2      accepted by the second judge alone (judge_groq.parquet: gpt-oss-120b)
  either  accepted by at least one judge
  both    accepted by both judges (the paper's filter)
and reports, per filter: the CF range of near-best extractors (standard pool, seed-mean, test dec within .01 of the
best), the effect of selecting by Conflict instead of Dec on CF (mean over samples, and the share of samples where
CF falls), with the erased probe in the pool, and the rank correlation of CF with the 'both' filter over competitive
extractors. -> results/rev_I_filter_robustness.csv, results/rev_I_summary.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from dv import validity as V  # noqa: E402
from dv.datasets import ALL as DATASETS  # noqa: E402
from dv.judging import accept  # noqa: E402

MODELS = ["qwen7b", "gemma9b"]
FILTERS = ["all", "j1", "j2", "either", "both"]


def verdicts(ds, pr, f):
    j = pd.read_parquet(ROOT / f"data/{ds}/{f}")
    acc = pd.Series(accept(j).values, index=pd.MultiIndex.from_frame(j[["uid_a", "uid_b"]]))
    return acc.reindex(pd.MultiIndex.from_frame(pr[["uid_a", "uid_b"]])).fillna(False).values.astype(bool)


def cf_by_filter(ds, mo):
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet").reset_index(drop=True)
    z = np.load(RES / f"scores_{ds}_{mo}.npz", allow_pickle=True)
    S, keys = z["S"].astype(np.float64), z["keys"]
    pos = {u: i for i, u in enumerate(z["uid"])}
    pr = pr[pr.uid_a.isin(pos) & pr.uid_b.isin(pos) & pr.kind.isin(["nuisance", "nuisance_rev"])].reset_index(drop=True)
    te = tx[(tx.split == "test") & tx["T"].notna() & tx["N"].notna()]
    ti = np.array([pos[u] for u in te.uid])
    t, n = te["T"].values.astype(int), te["N"].values.astype(int)
    w = V.cell_weights(t, n, 0.5)
    St = S[:, ti]
    dT = (St * (w * (t == 1))).sum(1) / (w * (t == 1)).sum() - (St * (w * (t == 0))).sum(1) / (w * (t == 0)).sum()
    D = (S[:, [pos[u] for u in pr.uid_b]] - S[:, [pos[u] for u in pr.uid_a]]) * pr.sign.values[None, :]
    j1, j2 = verdicts(ds, pr, "judge.parquet"), verdicts(ds, pr, "judge_groq.parquet")
    masks = {"all": np.ones(len(pr), bool), "j1": j1, "j2": j2, "either": j1 | j2, "both": j1 & j2}
    out = pd.DataFrame({"key": keys})
    for f, m in masks.items():
        parts = [D[:, m & (pr.kind == k).values].mean(1) / dT for k in ["nuisance", "nuisance_rev"]
                 if (m & (pr.kind == k).values).sum() >= 10]
        out[f"CF_{f}"] = np.mean(parts, axis=0)
        out[f"n_{f}"] = int(m.sum())
    return out


rows = []
for ds in DATASETS:
    for mo in MODELS:
        p = pd.read_parquet(RES / f"pop_{ds}_{mo}.parquet")
        p["key"] = p.layer.astype(str) + "|" + p.rho.astype(str) + "|" + p.seed.astype(str) + "|" + p.extractor
        rows.append(p.merge(cf_by_filter(ds, mo), on="key").assign(dataset=ds, model=mo))
pop = pd.concat(rows, ignore_index=True)
pop = pop[pop.extractor != "pls_k1"]
CFS = [f"CF_{f}" for f in FILTERS]
mp = pop.groupby(["dataset", "model", "rho", "layer", "extractor", "family"], as_index=False)[
    ["test_dec", "dev_dec", "dev_conflict"] + CFS].mean()

out = []
for (ds, mo, rho), g0 in mp.groupby(["dataset", "model", "rho"]):
    g = g0[g0.family != "ne_lr"]
    near = g[g.test_dec >= g.test_dec.max() - 0.01]
    comp = g[g.test_dec >= g.test_dec.max() - 0.05]
    for f in FILTERS:
        r = dict(dataset=ds, model=mo, rho=rho, filter=f, n_pairs=int(pop[(pop.dataset == ds) & (pop.model == mo)][f"n_{f}"].iloc[0]),
                 near_min=near[f"CF_{f}"].min(), near_max=near[f"CF_{f}"].max(),
                 spearman_vs_both=spearmanr(comp[f"CF_{f}"], comp.CF_both).statistic)
        # selection: max dev conflict vs max dev dec, per (rho, seed) sample, standard and full pools
        for pool, keep in [("std", lambda x: x[x.family != "ne_lr"]), ("full", lambda x: x)]:
            d = []
            for _, s0 in pop[(pop.dataset == ds) & (pop.model == mo) & (pop.rho == rho)].groupby("seed"):
                s = keep(s0)
                d.append(s.loc[s.dev_conflict.idxmax(), f"CF_{f}"] - s.loc[s.dev_dec.idxmax(), f"CF_{f}"])
            r[f"selconf_dCF_{pool}"], r[f"selconf_share_lower_{pool}"] = np.mean(d), np.mean(np.array(d) < 0)
        out.append(r)
R = pd.DataFrame(out)
R.to_csv(RES / "rev_I_filter_robustness.csv", index=False)

# compact summary: does each headline conclusion hold under every filter?
S = []
for (ds, rho), g in R[R.rho.isin([0.5, 0.9])].groupby(["dataset", "rho"]):
    for f, h in g.groupby("filter"):
        S.append(dict(dataset=ds, rho=rho, filter=f, n_pairs=" / ".join(str(x) for x in h.n_pairs),
                      near_range=" ; ".join(f"{a:.2f}..{b:.2f}" for a, b in zip(h.near_min, h.near_max)),
                      spearman_vs_both=h.spearman_vs_both.min(),
                      selconf_dCF_full=" / ".join(f"{x:+.3f}" for x in h.selconf_dCF_full),
                      selconf_dCF_std=" / ".join(f"{x:+.3f}" for x in h.selconf_dCF_std)))
S = pd.DataFrame(S)
S.to_csv(RES / "rev_I_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.to_string(index=False))
