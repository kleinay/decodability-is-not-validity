"""Compact paper tables -> results/table_main.csv, results/table_selection.csv, results/key_numbers.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parents[1] / "results"
pop = pd.concat([pd.read_parquet(p) for p in sorted(RES.glob("pop_*.parquet"))], ignore_index=True)
num = [c for c in pop.columns if c.startswith(("test_", "dev_", "cf_"))]
mp = pop.groupby(["dataset", "model", "rho", "layer", "extractor", "family"], as_index=False)[num].mean()

rows = []
for (ds, mo, rho), g in mp.groupby(["dataset", "model", "rho"]):
    best = g.test_dec.max()
    comp = g[g.test_dec >= best - 0.05]
    near = g[g.test_dec >= best - 0.01]
    rows.append(dict(dataset=ds, model=mo, rho=rho, best_dec=best, n_near=len(near), n_comp=len(comp),
                     cf_min=near.cf_nuisance.min(), cf_max=near.cf_nuisance.max(),
                     conf_min=near.test_conflict.min(), conf_max=near.test_conflict.max(),
                     leak_min=near.test_leak.min(), leak_max=near.test_leak.max(),
                     sp_dec_cf_comp=spearmanr(comp.test_dec, comp.cf_nuisance).statistic,
                     sp_dec_conf_comp=spearmanr(comp.test_dec, comp.test_conflict).statistic,
                     sp_leak_cf_all=spearmanr(g.test_leak, g.cf_nuisance).statistic,
                     cf_target_min=near.cf_target.min() if "cf_target" in near and near.cf_target.notna().any() else np.nan,
                     cf_target_max=near.cf_target.max() if "cf_target" in near and near.cf_target.notna().any() else np.nan))
main = pd.DataFrame(rows)
main.to_csv(RES / "table_main.csv", index=False)

sel = pd.read_csv(RES / "selection_by_seed.csv")
piv = sel.pivot_table(index=["dataset", "model", "rho", "seed"], columns="criterion",
                      values=["test_dec", "test_conflict", "test_leak", "cf_nuisance"])
d = pd.DataFrame({m: piv[(m, "dev_conflict")] - piv[(m, "dev_dec")] for m in ["test_dec", "test_conflict", "test_leak", "cf_nuisance"]}).reset_index()
agg = d.groupby(["dataset", "model", "rho"]).agg(["mean", "min", "max"])
agg.columns = ["_".join(c) for c in agg.columns]
base = sel[sel.criterion == "dev_dec"].groupby(["dataset", "model", "rho"])[["test_dec", "test_conflict", "cf_nuisance"]].mean().add_prefix("decsel_")
alt = sel[sel.criterion == "dev_conflict"].groupby(["dataset", "model", "rho"])[["test_dec", "test_conflict", "cf_nuisance"]].mean().add_prefix("valsel_")
selt = pd.concat([base, alt, agg], axis=1).reset_index()
selt.to_csv(RES / "table_selection.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(main.round(3).to_string()); print(selt.round(3).to_string())

# natural (unresampled) T-N agreement in each training pool, to anchor what rho values are realistic
nat = {}
from datasets import load_dataset  # noqa: E402
b = load_dataset("LabHC/bias_in_bios")["train"].to_pandas()
b = b[b.profession.isin([25, 13])]
nat["bios_P(male|surgeon)"] = float((b[b.profession == 25].gender == 0).mean())
nat["bios_P(female|nurse)"] = float((b[b.profession == 13].gender == 1).mean())
c = load_dataset("CEBaB/CEBaB")["train_observational"].to_pandas()
m = {"Positive": 1, "Negative": 0}
c = c.assign(S=c.service_aspect_majority.map(m), F=c.food_aspect_majority.map(m)).dropna(subset=["S", "F"])
nat["cebab_obs_P(food+|service+)"] = float(c[c.S == 1].F.mean())
nat["cebab_obs_P(food-|service-)"] = float(1 - c[c.S == 0].F.mean())
w = load_dataset("pietrolesci/civilcomments-wilds")["train"].to_pandas()
nid = (w.muslim == 1) | (w.LGBTQ == 1)
nat["civil_P(toxic|identity)"] = float(w[nid].toxicity.mean())
nat["civil_P(toxic|no identity)"] = float(w[~nid].toxicity.mean())
json.dump(nat, open(RES / "natural_correlations.json", "w"), indent=1)
print(nat)

# ---- robustness: standard families only (no nuisance labels used in fitting)
std = pop[pop.family != "ne_lr"]
mps = std.groupby(["dataset", "model", "rho", "layer", "extractor", "family"], as_index=False)[num].mean()
rows = []
for (ds, mo, rho), g in mps.groupby(["dataset", "model", "rho"]):
    best = g.test_dec.max()
    near = g[g.test_dec >= best - 0.01]
    i, j = np.triu_indices(len(g), 1)
    tied = np.abs(g.test_dec.values[i] - g.test_dec.values[j]) < 0.005
    dcf = np.abs(g.cf_nuisance.values[i] - g.cf_nuisance.values[j])[tied]
    rows.append(dict(dataset=ds, model=mo, rho=rho, best_dec=best, n=len(g), n_near=len(near),
                     cf_min=near.cf_nuisance.min(), cf_max=near.cf_nuisance.max(),
                     conf_min=near.test_conflict.min(), conf_max=near.test_conflict.max(),
                     tied_pairs=int(tied.sum()), frac_tied_cf_gt01=float((dcf > 0.1).mean())))
pd.DataFrame(rows).to_csv(RES / "table_main_std.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())

out = []
for (ds, mo, rho, seed), g in std.groupby(["dataset", "model", "rho", "seed"]):
    a, b = g.loc[g.dev_dec.idxmax()], g.loc[g.dev_conflict.idxmax()]
    out.append(dict(dataset=ds, model=mo, rho=rho, seed=seed, cf_decsel=a.cf_nuisance, cf_valsel=b.cf_nuisance,
                    conf_decsel=a.test_conflict, conf_valsel=b.test_conflict,
                    dec_decsel=a.test_dec, dec_valsel=b.test_dec, decbal_decsel=a.test_dec_bal, decbal_valsel=b.test_dec_bal,
                    ext_valsel=f"{b.extractor}@{b.layer}", ext_decsel=f"{a.extractor}@{a.layer}"))
o = pd.DataFrame(out)
o.to_csv(RES / "selection_std_by_seed.csv", index=False)
o["d_cf"] = o.cf_valsel - o.cf_decsel
o["d_dec"] = o.dec_valsel - o.dec_decsel
o["d_conf"] = o.conf_valsel - o.conf_decsel
s = o.groupby(["dataset", "model", "rho"]).agg(cf_decsel=("cf_decsel", "mean"), cf_valsel=("cf_valsel", "mean"),
                                               d_conf=("d_conf", "mean"), d_dec=("d_dec", "mean"),
                                               d_cf_max=("d_cf", "max"), d_conf_min=("d_conf", "min")).reset_index()
s.to_csv(RES / "table_selection_std.csv", index=False)
print(s.round(3).to_string())
