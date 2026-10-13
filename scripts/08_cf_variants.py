"""Counterfactual-sensitivity variants for every fitted extractor, computed from the saved raw scores
(results/scores_<ds>_<model>.npz) -> results/cfv_<ds>_<model>.parquet, one row per extractor key.

For each pair set P (edit kind x subset) and score s:
  cf        signed mean of sign*(s_b - s_a), / Delta_T           (as in the paper; 0 does NOT imply invariance)
  cfabs     mean |s_b - s_a|, / Delta_T                          (magnitude, no cancellation)
  cfsd      signed mean / pooled within-(T,N)-cell SD of s       (alternative normalization)
  cong      share of pairs moving in the nuisance-congruent direction
Pair subsets: all; 'val' = accepted by every available LLM judge; 'len' = |Δtokens| <= 2;
per-direction for bidirectional kinds ('pos': edit moves N toward its T=1-associated value; 'neg': away).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dv import validity as V  # noqa: E402
from dv.judging import accept  # noqa: E402


def judge_ok(ds, pr):
    """Boolean per pair: accepted by every judge file present (missing verdicts count as not accepted)."""
    ok = pd.Series(True, index=pr.index)
    n_j = 0
    for f in ["judge.parquet", "judge_groq.parquet"]:
        p = ROOT / f"data/{ds}/{f}"
        if not p.exists():
            continue
        n_j += 1
        j = pd.read_parquet(p)
        acc = accept(j).values
        acc = pd.Series(acc, index=pd.MultiIndex.from_frame(j[["uid_a", "uid_b"]]))
        key = pd.MultiIndex.from_frame(pr[["uid_a", "uid_b"]])
        ok &= pd.Series(acc.reindex(key).fillna(False).values, index=pr.index).astype(bool)
    return ok if n_j else pd.Series(np.nan, index=pr.index), n_j


def main(ds, model):
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet").reset_index(drop=True)
    z = np.load(ROOT / f"results/scores_{ds}_{model}.npz", allow_pickle=True)
    S, keys = z["S"].astype(np.float64), z["keys"]
    pos = {u: i for i, u in enumerate(z["uid"])}
    pr = pr[pr.uid_a.isin(pos) & pr.uid_b.isin(pos)].reset_index(drop=True)
    ntok = pd.read_parquet(ROOT / f"data/{ds}/ntok_qwen.parquet").ntok if (ROOT / f"data/{ds}/ntok_qwen.parquet").exists() else None
    te = tx[(tx.split == "test") & tx["T"].notna() & tx["N"].notna()]
    ti = np.array([pos[u] for u in te.uid])
    t, n = te["T"].values.astype(int), te["N"].values.astype(int)
    w = V.cell_weights(t, n, 0.5)
    St = S[:, ti]
    dT = (St * (w * (t == 1))).sum(1) / (w * (t == 1)).sum() - (St * (w * (t == 0))).sum(1) / (w * (t == 0)).sum()
    # pooled within-cell SD (equal cell weights)
    var = np.zeros(len(S))
    for ct in (0, 1):
        for cn in (0, 1):
            m = (t == ct) & (n == cn)
            var += St[:, m].var(1) / 4
    sd = np.sqrt(var)
    ok, n_judges = judge_ok(ds, pr)
    if ntok is not None:
        dlen = np.abs(ntok.reindex(pr.uid_b).values - ntok.reindex(pr.uid_a).values)
    else:
        dlen = np.zeros(len(pr))
    A = np.array([pos[u] for u in pr.uid_a])
    Bi = np.array([pos[u] for u in pr.uid_b])
    D = (S[:, Bi] - S[:, A]) * pr.sign.values[None, :]  # oriented score change per pair
    out = pd.DataFrame({"key": keys, "deltaT": dT, "sd_within": sd})
    subsets = {"all": np.ones(len(pr), bool), "len": dlen <= 2}
    if n_judges:
        subsets["val"] = ok.values.astype(bool)
    for kind in pr.kind.unique():
        km = (pr.kind == kind).values
        for sname, sm in subsets.items():
            m = km & sm
            if m.sum() < 10:
                continue
            tag = f"{kind}_{sname}"
            out[f"cf_{tag}"] = D[:, m].mean(1) / dT
            out[f"cfabs_{tag}"] = np.abs(D[:, m]).mean(1) / dT
            out[f"cfsd_{tag}"] = D[:, m].mean(1) / sd
            out[f"cong_{tag}"] = (D[:, m] > 0).mean(1)
            out[f"n_{tag}"] = int(m.sum())
        if kind in ("nuisance", "target") and (pr.sign[km] == -1).any() and (pr.sign[km] == 1).any():
            for dname, dv in [("pos", 1), ("neg", -1)]:
                m = km & (pr.sign.values == dv)
                out[f"cf_{kind}_dir{dname}"] = D[:, m].mean(1) / dT
                out[f"n_{kind}_dir{dname}"] = int(m.sum())
    # Civil: two-sided estimate = mean of the two directions (forward Muslim/LGBTQ->Christian/straight, reverse)
    for sname in subsets:
        a, b = f"cf_nuisance_{sname}", f"cf_nuisance_rev_{sname}"
        if a in out and b in out:
            out[f"cf2_{sname}"] = 0.5 * (out[a] + out[b])
            out[f"cfabs2_{sname}"] = 0.5 * (out[f"cfabs_nuisance_{sname}"] + out[f"cfabs_nuisance_rev_{sname}"])
        elif a in out:
            out[f"cf2_{sname}"], out[f"cfabs2_{sname}"] = out[a], out[f"cfabs_nuisance_{sname}"]
    kk = out.key.str.split("|", expand=True)
    out["layer"], out["rho"], out["seed"], out["extractor"] = kk[0].astype(int), kk[1].astype(float), kk[2].astype(int), kk[3]
    out.insert(0, "model", model)
    out.insert(0, "dataset", ds)
    out.to_parquet(ROOT / f"results/cfv_{ds}_{model}.parquet")
    print(ds, model, "judges:", n_judges, "| pair counts:",
          {c[2:]: int(out[c].iloc[0]) for c in out.columns if c.startswith("n_")})


if __name__ == "__main__":
    for ds in ["cebab", "bios", "civil"]:
        for model in sys.argv[1:] or ["qwen7b", "gemma9b"]:
            if (ROOT / f"results/scores_{ds}_{model}.npz").exists():
                main(ds, model)
