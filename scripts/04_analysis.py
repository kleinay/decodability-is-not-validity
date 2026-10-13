"""All paper statistics from results/pop_*.parquet + results/scores_*.npz -> results/*.csv, results/key_numbers.json

 (1) population summary: per (dataset, model, rho), rank correlation of decodability with each validity facet,
     and the spread of validity among the decodability-near-best extractors
 (2) tied pairs: among extractor pairs whose decodability differs by < 0.005, how often validity differs materially
 (3) selection: pick the extractor by dev decodability (standard practice) vs. by dev conflict AUROC
 (4) pre-specified family contrast with bootstrap CIs: each family's (layer, hyperparameter) is chosen by dev
     decodability on the seed-0 training sample; test-set differences between families, paired bootstrap
 (5) observational vs. counterfactual: does conditional leakage predict counterfactual nuisance sensitivity?
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dv import validity as V  # noqa: E402

RES = ROOT / "results"
VAL = ["test_conflict", "test_leak", "cf_nuisance"]
NEAR = 0.01
TIE = 0.005
B = 2000

pops = [pd.read_parquet(p) for p in sorted(RES.glob("pop_*.parquet"))]
pop = pd.concat(pops, ignore_index=True)
key = ["dataset", "model", "rho"]
cfg = ["layer", "extractor", "family"]
num = [c for c in pop.columns if c.startswith(("test_", "dev_", "cf_"))]
mean_pop = pop.groupby(key + cfg, as_index=False)[num].mean()  # average over the 3 training resamples
kn = {}

# ---------------------------------------------------------------- (1) population summary
rows = []
for k, g in mean_pop.groupby(key):
    r = dict(zip(key, k), n_extractors=len(g))
    best = g.test_dec.max()
    near = g[g.test_dec >= best - NEAR]
    r["best_dec"], r["n_near"] = best, len(near)
    for v in VAL:
        r[f"rho_dec_{v}"] = spearmanr(g.test_dec, g[v]).statistic
        r[f"near_min_{v}"], r[f"near_max_{v}"] = near[v].min(), near[v].max()
    r["near_families"] = ",".join(sorted(near.family.unique()))
    rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(RES / "population_summary.csv", index=False)
print(summ.round(3).to_string())

# ---------------------------------------------------------------- (2) tied pairs
rows = []
for k, g in mean_pop.groupby(key):
    d = g.test_dec.values
    i, j = np.triu_indices(len(g), 1)
    tied = np.abs(d[i] - d[j]) < TIE
    r = dict(zip(key, k), n_pairs=int(tied.sum()))
    for v, thr in [("test_conflict", 0.05), ("test_leak", 0.05), ("cf_nuisance", 0.10)]:
        x = g[v].values
        diff = np.abs(x[i] - x[j])[tied]
        r[f"frac_{v}_gt{thr}"] = float((diff > thr).mean()) if len(diff) else np.nan
        r[f"max_{v}"] = float(diff.max()) if len(diff) else np.nan
    rows.append(r)
tied_df = pd.DataFrame(rows)
tied_df.to_csv(RES / "tied_pairs.csv", index=False)
print(tied_df.round(3).to_string())

# ---------------------------------------------------------------- (3) selection
rows = []
for k, g in pop.groupby(key + ["seed"]):
    for crit in ["dev_dec", "dev_conflict"]:
        s = g.loc[g[crit].idxmax()]
        rows.append(dict(zip(key + ["seed"], k), criterion=crit, extractor=s.extractor, layer=s.layer,
                         **{c: s[c] for c in ["test_dec", "test_dec_bal"] + VAL}))
sel = pd.DataFrame(rows)
sel_mean = sel.groupby(key + ["criterion"], as_index=False)[["test_dec", "test_dec_bal"] + VAL].mean()
sel.to_csv(RES / "selection_by_seed.csv", index=False)
sel_mean.to_csv(RES / "selection.csv", index=False)
print(sel_mean.round(3).to_string())

# ---------------------------------------------------------------- (4) pre-specified family contrasts (bootstrap)
def load_scores(ds, model):
    z = np.load(RES / f"scores_{ds}_{model}.npz", allow_pickle=True)
    return dict(zip(z["keys"], z["S"])), {u: i for i, u in enumerate(z["uid"])}


def metrics_from(s, idx, ev, pairs, rho):
    """All metrics for one score vector given index arrays into it (bootstrap-resampled)."""
    t, n = ev
    tt, nn = t[idx["test"]], n[idx["test"]]
    st = s[idx["test_pos"]]
    out = {"dec": V.dec_auroc(st, tt, nn, rho), "conflict": V.conflict_auroc(st, tt, nn), "leak": V.leak(st, tt, nn)}
    dT = V.delta_t(st, tt, nn)
    for kind, (a, b, sg) in pairs.items():
        sel = idx[f"pair_{kind}"]
        out[f"cf_{kind}"] = V.cf_effect(s[a[sel]], s[b[sel]], sg[sel], dT)
    if "cf_nuisance_rev" in out:  # two-sided Civil estimate, as in 03_fit_population.py
        out["cf_nuisance"] = 0.5 * (out["cf_nuisance"] + out["cf_nuisance_rev"])
    return out


contrast_rows = []
for (ds, model), g0 in pop[pop.seed == 0].groupby(["dataset", "model"]):
    texts = pd.read_parquet(ROOT / "data" / ds / "texts.parquet").set_index("uid")
    pairs_df = pd.read_parquet(ROOT / "data" / ds / "pairs.parquet")
    S, pos = load_scores(ds, model)
    test = texts[(texts.split == "test") & texts["T"].notna() & texts["N"].notna()]
    test_pos = np.array([pos[u] for u in test.index])
    t, n = test["T"].values.astype(int), test["N"].values.astype(int)
    pairs = {kind: (np.array([pos[u] for u in g.uid_a]), np.array([pos[u] for u in g.uid_b]), g.sign.values.astype(float))
             for kind, g in pairs_df.groupby("kind")}
    rng = np.random.RandomState(0)
    for rho, g in g0.groupby("rho"):
        chosen = g.loc[g.groupby("family").dev_dec.idxmax()]
        fams = list(chosen.family)
        keys = {r.family: f"{r.layer}|{r.rho}|{r.seed}|{r.extractor}" for r in chosen.itertuples()}
        base_idx = {"test": np.arange(len(t)), "test_pos": test_pos,
                    **{f"pair_{k}": np.arange(len(p[0])) for k, p in pairs.items()}}
        point = {f: metrics_from(S[keys[f]], base_idx, (t, n), pairs, rho) for f in fams}
        cells = [np.where((t == a) & (n == b))[0] for a in (0, 1) for b in (0, 1)]
        boots = {f: [] for f in fams}
        for _ in range(B):
            ti = np.concatenate([rng.choice(c, len(c)) for c in cells])
            bi = {"test": ti, "test_pos": test_pos[ti],
                  **{f"pair_{k}": rng.randint(0, len(p[0]), len(p[0])) for k, p in pairs.items()}}
            for f in fams:
                boots[f].append(metrics_from(S[keys[f]], bi, (t, n), pairs, rho))
        for f in fams:
            row = dict(dataset=ds, model=model, rho=rho, family=f, extractor=keys[f].split("|")[3],
                       layer=int(keys[f].split("|")[0]), contrast="point")
            for m, val in point[f].items():
                arr = np.array([b[m] for b in boots[f]])
                row[m], row[m + "_lo"], row[m + "_hi"] = val, *np.percentile(arr, [2.5, 97.5])
            contrast_rows.append(row)
        for f1, f2 in itertools.combinations(fams, 2):
            row = dict(dataset=ds, model=model, rho=rho, family=f"{f1}-{f2}", contrast="diff")
            for m in point[f1]:
                arr = np.array([a[m] - b[m] for a, b in zip(boots[f1], boots[f2])])
                row[m], row[m + "_lo"], row[m + "_hi"] = point[f1][m] - point[f2][m], *np.percentile(arr, [2.5, 97.5])
            contrast_rows.append(row)
    print("contrasts done", ds, model, flush=True)
con = pd.DataFrame(contrast_rows)
con.to_csv(RES / "family_contrasts.csv", index=False)

# ---------------------------------------------------------------- (5) observational vs counterfactual
rows = []
for k, g in mean_pop.groupby(key):
    rows.append(dict(zip(key, k),
                     spearman_leak_cf=spearmanr(g.test_leak, g.cf_nuisance).statistic,
                     spearman_conflict_cf=spearmanr(g.test_conflict, g.cf_nuisance).statistic,
                     spearman_dec_cf=spearmanr(g.test_dec, g.cf_nuisance).statistic))
obs = pd.DataFrame(rows)
obs.to_csv(RES / "obs_vs_cf.csv", index=False)
print(obs.round(3).to_string())
