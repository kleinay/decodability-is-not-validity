"""Analyses added in response to the critical review -> results/rev_*.csv and results/rev_key_numbers.json

Primary nuisance-sensitivity metric: CF = signed CF_N on judge-validated pairs (Civil: mean of both edit
directions), with |CF| (mean absolute change) alongside. pls_k1 is dropped everywhere as an exact duplicate of
dom (07_audit.py). 'standard' = dom/lr/pls (no nuisance labels); 'full' adds the nuisance-erased probe.

 A  near-best sets: defined on TEST dec (descriptive) and on DEV dec (what a practitioner can see)
 B  equivalence: paired-bootstrap TOST (margin 0.01 AUROC, 90% CI) of each near-best extractor vs the best
 C  rank correlations of dec with each facet, with bootstrap-over-extractors 95% CIs, every setting
 D  selection: four dev criteria (max dec; max conflict; min |leak| s.t. dec >= best-0.01; max dec-|leak|)
 E  fixed-method comparison (probe vs nuisance-erased probe, dev-chosen) with TOST and cluster bootstrap
 F  geometry: does observational within-class nuisance geometry predict diff-of-means CF at rho=0.5?
 G  edit validation: judge acceptance, inter-judge agreement, CF on accepted vs rejected pairs, placebo floors
 H  worked example (CEBaB review with food and service edits; two accuracy-matched extractors)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from dv import validity as V  # noqa: E402
from dv.judging import accept  # noqa: E402

MODELS = sys.argv[1:] or ["qwen7b", "gemma9b"]
KN = {}
rng = np.random.RandomState(0)


# ---------------------------------------------------------------- load & merge
def load():
    rows = []
    for ds in ["cebab", "bios", "civil"]:
        for mo in MODELS:
            p = pd.read_parquet(RES / f"pop_{ds}_{mo}.parquet")
            p["key"] = p.layer.astype(str) + "|" + p.rho.astype(str) + "|" + p.seed.astype(str) + "|" + p.extractor
            v = pd.read_parquet(RES / f"cfv_{ds}_{mo}.parquet").drop(columns=["dataset", "model", "layer", "rho", "seed", "extractor"])
            rows.append(p.merge(v, on="key"))
    d = pd.concat(rows, ignore_index=True)
    d = d[d.extractor != "pls_k1"].copy()
    sub = "val" if "cf2_val" in d and d.cf2_val.notna().any() else "all"
    d["CF"], d["CFABS"] = d[f"cf2_{sub}"], d[f"cfabs2_{sub}"]
    d["PLACEBO_ABS"] = d.get(f"cfabs_placebo_{sub}", np.nan)
    KN["cf_pair_subset"] = sub
    return d


pop = load()
num = [c for c in pop.columns if pop[c].dtype.kind == "f" and c not in ("rho",)]
mp = pop.groupby(["dataset", "model", "rho", "layer", "extractor", "family"], as_index=False)[num].mean()
pools = {"standard": lambda g: g[g.family != "ne_lr"], "full": lambda g: g}
FACETS = ["CF", "CFABS", "test_conflict", "test_leak_abs"]

# ---------------------------------------------------------------- A near-best
rows = []
for (ds, mo, rho), g0 in mp.groupby(["dataset", "model", "rho"]):
    for pool, f in pools.items():
        g = f(g0)
        for basis in ["test_dec", "dev_dec"]:
            near = g[g[basis] >= g[basis].max() - 0.01]
            r = dict(dataset=ds, model=mo, rho=rho, pool=pool, defined_on=basis, n=len(g), n_near=len(near),
                     best_test_dec=g.test_dec.max(), near_test_dec_min=near.test_dec.min())
            for c in FACETS + ["PLACEBO_ABS"]:
                r[f"{c}_min"], r[f"{c}_max"] = near[c].min(), near[c].max()
            rows.append(r)
A = pd.DataFrame(rows)
A.to_csv(RES / "rev_A_nearbest.csv", index=False)


# ---------------------------------------------------------------- shared: per-setting score access
def setting_scores(ds, mo):
    z = np.load(RES / f"scores_{ds}_{mo}.npz", allow_pickle=True)
    S = dict(zip(z["keys"], z["S"]))
    pos = {u: i for i, u in enumerate(z["uid"])}
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    te = tx[(tx.split == "test") & tx["T"].notna() & tx["N"].notna()]
    ti = np.array([pos[u] for u in te.uid])
    t, n = te["T"].values.astype(int), te["N"].values.astype(int)
    if ds == "cebab":
        fam = pd.read_parquet(ROOT / "data/cebab/family.parquet").set_index("uid").family
        cl = pd.factorize(fam.reindex(te.uid).values)[0]
    else:
        cl = np.arange(len(t))
    return S, pos, ti, t, n, cl


def boot_index(t, n, cl, B):
    """Cluster bootstrap (CEBaB: review family) stratified by (T,N) cell for non-clustered data."""
    out = []
    ucl = np.unique(cl)
    members = {c: np.where(cl == c)[0] for c in ucl}
    for _ in range(B):
        if len(ucl) < len(t):  # clustered
            pick = rng.choice(ucl, len(ucl))
            out.append(np.concatenate([members[c] for c in pick]))
        else:
            out.append(np.concatenate([rng.choice(np.where((t == a) & (n == b))[0], ((t == a) & (n == b)).sum())
                                       for a in (0, 1) for b in (0, 1)]))
    return out


def dec_of(s, t, n, rho):
    return V.dec_auroc(s, t, n, rho)


# ---------------------------------------------------------------- B equivalence (seed 0)
rows = []
B_EQ = 300
for (ds, mo, rho), g0 in pop[(pop.seed == 0) & pop.rho.isin([0.5, 0.9])].groupby(["dataset", "model", "rho"]):
    S, pos, ti, t, n, cl = setting_scores(ds, mo)
    idx = boot_index(t, n, cl, B_EQ)
    for pool, f in pools.items():
        g = f(g0)
        best = g.loc[g.test_dec.idxmax()]
        near = g[g.test_dec >= best.test_dec - 0.01]
        sb = S[best.key][ti].astype(float)
        bb = np.array([dec_of(sb[i], t[i], n[i], rho) for i in idx])
        eq = []
        for _, r in near.iterrows():
            if r.key == best.key:
                eq.append(True)
                continue
            s = S[r.key][ti].astype(float)
            d = bb - np.array([dec_of(s[i], t[i], n[i], rho) for i in idx])
            lo, hi = np.percentile(d, [5, 95])
            eq.append(bool(lo > -0.01 and hi < 0.01))
        e = near[np.array(eq)]
        rows.append(dict(dataset=ds, model=mo, rho=rho, pool=pool, n_near=len(near), n_equiv=len(e),
                         CF_min=e.CF.min(), CF_max=e.CF.max(), CFABS_min=e.CFABS.min(), CFABS_max=e.CFABS.max(),
                         conflict_min=e.test_conflict.min(), conflict_max=e.test_conflict.max()))
    print("equivalence", ds, mo, rho, flush=True)
Bq = pd.DataFrame(rows)
Bq.to_csv(RES / "rev_B_equivalence.csv", index=False)

# ---------------------------------------------------------------- C correlations with CIs
rows = []
for (ds, mo, rho), g0 in mp.groupby(["dataset", "model", "rho"]):
    for pool, f in pools.items():
        g = f(g0)
        g = g[g.test_dec >= g.test_dec.max() - 0.05]  # competitive extractors (CF unstable otherwise)
        r = dict(dataset=ds, model=mo, rho=rho, pool=pool, n=len(g))
        for c in FACETS:
            x, y = g.test_dec.values, g[c].values
            bs = [spearmanr(x[i], y[i]).statistic for i in (rng.randint(0, len(g), len(g)) for _ in range(1000))]
            r[f"rho_{c}"] = spearmanr(x, y).statistic
            r[f"rho_{c}_lo"], r[f"rho_{c}_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(r)
C = pd.DataFrame(rows)
C.to_csv(RES / "rev_C_correlations.csv", index=False)

# ---------------------------------------------------------------- D selection criteria
CRIT = {
    "max_dec": lambda g: g.dev_dec.idxmax(),
    "max_conflict": lambda g: g.dev_conflict.idxmax(),
    "min_leak_at_dec": lambda g: g[g.dev_dec >= g.dev_dec.max() - 0.01].dev_leak_abs.idxmin(),
    "max_dec_minus_leak": lambda g: (g.dev_dec - g.dev_leak_abs).idxmax(),
}
rows = []
for (ds, mo, rho, seed), g0 in pop.groupby(["dataset", "model", "rho", "seed"]):
    for pool, f in pools.items():
        g = f(g0)
        for cname, crit in CRIT.items():
            s = g.loc[crit(g)]
            rows.append(dict(dataset=ds, model=mo, rho=rho, seed=seed, pool=pool, criterion=cname, extractor=s.extractor,
                             layer=s.layer, **{c: s[c] for c in ["test_dec", "test_dec_bal", "test_conflict",
                                                                 "test_leak_abs", "CF", "CFABS", "PLACEBO_ABS"]}))
D = pd.DataFrame(rows)
D.to_csv(RES / "rev_D_selection_by_seed.csv", index=False)
base = D[D.criterion == "max_dec"].set_index(["dataset", "model", "rho", "seed", "pool"])
drows = []
for cname in CRIT:
    if cname == "max_dec":
        continue
    alt = D[D.criterion == cname].set_index(["dataset", "model", "rho", "seed", "pool"])
    diff = (alt[["test_dec", "test_dec_bal", "test_conflict", "test_leak_abs", "CF", "CFABS"]]
            - base[["test_dec", "test_dec_bal", "test_conflict", "test_leak_abs", "CF", "CFABS"]]).reset_index()
    diff["criterion"] = cname
    drows.append(diff)
Dd = pd.concat(drows)
Dsum = Dd.groupby(["dataset", "model", "rho", "pool", "criterion"]).agg(
    d_dec=("test_dec", "mean"), d_dec_worst=("test_dec", "min"), d_conflict=("test_conflict", "mean"),
    d_leak_abs=("test_leak_abs", "mean"), d_CF=("CF", "mean"), d_CF_worst=("CF", "max"), d_CFABS=("CFABS", "mean"),
    d_CFABS_worst=("CFABS", "max")).reset_index()
Dmean = D.groupby(["dataset", "model", "rho", "pool", "criterion"])[["test_dec", "test_conflict", "test_leak_abs", "CF", "CFABS"]].mean().reset_index()
Dsum.to_csv(RES / "rev_D_selection_deltas.csv", index=False)
Dmean.to_csv(RES / "rev_D_selection_means.csv", index=False)

# ---------------------------------------------------------------- E fixed-method comparison (seed 0)
rows = []
for (ds, mo, rho), g0 in pop[(pop.seed == 0) & pop.rho.isin([0.5, 0.9])].groupby(["dataset", "model", "rho"]):
    S, pos, ti, t, n, cl = setting_scores(ds, mo)
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    ch = {fam: g0[g0.family == fam].loc[g0[g0.family == fam].dev_dec.idxmax()] for fam in ["lr", "ne_lr"]}
    s1, s2 = (S[ch[f].key][ti].astype(float) for f in ["lr", "ne_lr"])
    idx = boot_index(t, n, cl, 1000)
    d = np.array([dec_of(s1[i], t[i], n[i], rho) - dec_of(s2[i], t[i], n[i], rho) for i in idx])
    rows.append(dict(dataset=ds, model=mo, rho=rho, lr_ext=f"{ch["lr"].extractor}@{ch["lr"].layer}",
                     ne_ext=f"{ch["ne_lr"].extractor}@{ch["ne_lr"].layer}",
                     d_dec=ch["lr"].test_dec - ch["ne_lr"].test_dec, d_dec_ci90_lo=np.percentile(d, 5),
                     d_dec_ci90_hi=np.percentile(d, 95), equivalent_0p01=bool(np.percentile(d, 5) > -0.01 and np.percentile(d, 95) < 0.01),
                     CF_lr=ch["lr"].CF, CF_ne=ch["ne_lr"].CF, CFABS_lr=ch["lr"].CFABS, CFABS_ne=ch["ne_lr"].CFABS,
                     conflict_lr=ch["lr"].test_conflict, conflict_ne=ch["ne_lr"].test_conflict))
E = pd.DataFrame(rows)
E.to_csv(RES / "rev_E_fixed_pair.csv", index=False)

# ---------------------------------------------------------------- F geometry
rows = []
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    for mo in MODELS:
        A_ = np.load(ROOT / f"data/{ds}/acts_{mo}.npz", allow_pickle=True)
        trp = np.where((tx.split == "train") & tx["T"].notna() & tx["N"].notna())[0]
        t_all, n_all = tx["T"].values[trp].astype(int), tx["N"].values[trp].astype(int)
        sel = np.concatenate([rng.choice(np.where((t_all == a) & (n_all == b))[0], 500, replace=False)
                              for a in (0, 1) for b in (0, 1)])  # balanced, rho = 0.5
        tr, t, n = trp[sel], t_all[sel], n_all[sel]
        for li, layer in enumerate(A_["layers"]):
            X = A_["X"][tr, li]
            Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
            dT = Z[t == 1].mean(0) - Z[t == 0].mean(0)
            dN = np.mean([Z[(t == c) & (n == 1)].mean(0) - Z[(t == c) & (n == 0)].mean(0) for c in (0, 1)], 0)
            cos = dT @ dN / np.linalg.norm(dT) / np.linalg.norm(dN)
            pred = (dN @ dT) / (dT @ dT)  # DoM score shift for an N flip that moves z by dN, in units of the T-class gap
            obs = mp[(mp.dataset == ds) & (mp.model == mo) & (mp.rho == 0.5) & (mp.layer == layer) & (mp.extractor == "dom")]
            rows.append(dict(dataset=ds, model=mo, layer=int(layer), cos_T_N=cos, dom_cf_predicted=pred,
                             dom_cf_observed=float(obs.CF.iloc[0]) if len(obs) else np.nan))
F = pd.DataFrame(rows)
F.to_csv(RES / "rev_F_geometry.csv", index=False)
KN["geometry_spearman_pred_obs"] = spearmanr(F.dom_cf_predicted, F.dom_cf_observed, nan_policy="omit").statistic
KN["geometry_pearson_pred_obs"] = float(np.corrcoef(F.dropna().dom_cf_predicted, F.dropna().dom_cf_observed)[0, 1])
KN["geometry_cos_by_dataset"] = F.groupby("dataset").cos_T_N.agg(["min", "max"]).round(3).to_dict("index")

# ---------------------------------------------------------------- G edit validation
rows = []
for ds in ["cebab", "bios", "civil"]:
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    jl = {}
    for jn, fn in [("qwen3_32b", "judge.parquet"), ("gpt_oss_120b", "judge_groq.parquet")]:
        p = ROOT / f"data/{ds}/{fn}"
        if p.exists():
            jl[jn] = pd.read_parquet(p)
    for jn, j in jl.items():
        y = lambda c: j[c].astype(str).str.startswith("yes")  # noqa: E731
        for kind, g in j.groupby("kind"):
            gi = g.index
            rows.append(dict(dataset=ds, judge=jn, kind=kind, n=len(g),
                             target_same=y("target_same")[gi].mean(), nuis_changed=j.nuis_same[gi].astype(str).str.startswith("no").mean(),
                             edit_fluent=y("edit_fluent")[gi].mean(), src_fluent=y("src_fluent")[gi].mean(),
                             other_major=j.other_change[gi].astype(str).str.startswith("major").mean(),
                             fluency_drop=(j.edit_fluent[gi].astype(str).str.startswith("no") & j.src_fluent[gi].astype(str).str.startswith("yes")).mean(),
                             accepted=accept(j).loc[gi].mean()))
G = pd.DataFrame(rows)
G.to_csv(RES / "rev_G_judges.csv", index=False)
# agreement on the accept/reject decision where both judges rated the pair
agree = []
for ds in ["cebab", "bios", "civil"]:
    a, b = ROOT / f"data/{ds}/judge.parquet", ROOT / f"data/{ds}/judge_groq.parquet"
    if not (a.exists() and b.exists()):
        continue
    ja, jb = pd.read_parquet(a), pd.read_parquet(b)

    def acc(j):
        return pd.Series(accept(j).values, index=pd.MultiIndex.from_frame(j[["uid_a", "uid_b"]]))
    m = pd.concat([acc(ja).rename("a"), acc(jb).rename("b")], axis=1, join="inner")
    m["kind"] = ja.set_index(["uid_a", "uid_b"]).kind.reindex(m.index).values
    for kind, g in m.groupby("kind"):
        po = (g.a == g.b).mean()
        pe = g.a.mean() * g.b.mean() + (1 - g.a.mean()) * (1 - g.b.mean())
        agree.append(dict(dataset=ds, kind=kind, n=len(g), accept_qwen=g.a.mean(), accept_gptoss=g.b.mean(),
                          accept_both=(g.a & g.b).mean(), agreement=po, kappa=(po - pe) / (1 - pe) if pe < 1 else np.nan))
Gag = pd.DataFrame(agree)
Gag.to_csv(RES / "rev_G_agreement.csv", index=False)
# CF on all vs validated pairs, and placebo floors, for competitive extractors
rows = []
for (ds, mo, rho), g in mp.groupby(["dataset", "model", "rho"]):
    g = g[g.test_dec >= g.test_dec.max() - 0.05]
    r = dict(dataset=ds, model=mo, rho=rho, n=len(g))
    for c in ["cf2_all", "cf2_val", "cf2_len", "cfabs2_all", "cfabs2_val", "cfabs_placebo_all", "cfabs_placebo_val",
              "cf_placebo_all", "cfabs_placebo_group_all", "cf_placebo_group_all", "cf_nuisance_rev_all",
              "cf_nuisance_all", "cf_nuisance_dirpos", "cf_nuisance_dirneg", "cfsd_nuisance_all"]:
        if c in g:
            r[f"{c}_median"] = g[c].median()
    if "cf2_val" in g and g.cf2_val.notna().any():
        r["spearman_all_vs_val"] = spearmanr(g.cf2_all, g.cf2_val).statistic
    if "cf_nuisance_dirpos" in g:
        r["spearman_dirpos_vs_dirneg"] = spearmanr(g.cf_nuisance_dirpos, g.cf_nuisance_dirneg).statistic
    if "cf_nuisance_rev_all" in g:
        r["spearman_fwd_vs_rev"] = spearmanr(g.cf_nuisance_all, g.cf_nuisance_rev_all).statistic
    r["spearman_cf_vs_cfsd"] = spearmanr(g.cf_nuisance_all, g.cfsd_nuisance_all).statistic
    r["spearman_cf_vs_cfabs"] = spearmanr(g.CF, g.CFABS).statistic
    rows.append(r)
Gcf = pd.DataFrame(rows)
Gcf.to_csv(RES / "rev_G_cf_robustness.csv", index=False)

# ---------------------------------------------------------------- H worked example
tx = pd.read_parquet(ROOT / "data/cebab/texts.parquet").set_index("uid")
pr = pd.read_parquet(ROOT / "data/cebab/pairs.parquet")
fam = pd.read_parquet(ROOT / "data/cebab/family.parquet").set_index("uid").family
both = set(pr[pr.kind == "nuisance"].uid_a) & set(pr[pr.kind == "target"].uid_a)
ex_rows = []
for mo in MODELS:
    S, pos, ti, t, n, cl = setting_scores("cebab", mo)
    e = E[(E.dataset == "cebab") & (E.model == mo) & (E.rho == 0.9)].iloc[0]
    for lab, spec in [("probe", e["lr_ext"]), ("erased", e["ne_ext"])]:
        ext, layer = spec.split("@")
        k = f"{layer}|0.9|0|{ext}"
        s = S[k].astype(float)
        lo = s[ti][(t == 0)].mean()
        hi = s[ti][(t == 1)].mean()
        for u in sorted(both):
            fp = pr[(pr.uid_a == u) & (pr.kind == "nuisance")].iloc[0]
            sp = pr[(pr.uid_a == u) & (pr.kind == "target")].iloc[0]
            ex_rows.append(dict(model=mo, extractor=lab, src=u, food_edit=fp.uid_b, service_edit=sp.uid_b,
                                s_src=(s[pos[u]] - lo) / (hi - lo), s_food=(s[pos[fp.uid_b]] - lo) / (hi - lo),
                                s_service=(s[pos[sp.uid_b]] - lo) / (hi - lo), food_sign=fp.sign, service_sign=sp.sign))
X_ = pd.DataFrame(ex_rows)
X_.to_csv(RES / "rev_H_examples.csv", index=False)

KN["n_extractors_dedup_per_setting"] = int(mp.groupby(["dataset", "model", "rho"]).size().iloc[0])
json.dump(KN, open(RES / "rev_key_numbers.json", "w"), indent=1, default=float)
print(json.dumps(KN, indent=1, default=float))
