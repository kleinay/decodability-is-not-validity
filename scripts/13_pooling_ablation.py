"""Readout ablation: last non-special token instead of mean pooling (Qwen2.5-7B), same extractors, edits and
judge-accepted pairs. Compares the headline quantities with the mean-pooled population -> paper/tables/tab_pooling.tex"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sub = json.load(open(RES / "rev_key_numbers.json"))["cf_pair_subset"]
DSN = {"cebab": "CEBaB", "bios": "BiasBios", "civil": "Civil"}


def load(ds, mo):
    p = pd.read_parquet(RES / f"pop_{ds}_{mo}.parquet")
    p["key"] = p.layer.astype(str) + "|" + p.rho.astype(str) + "|" + p.seed.astype(str) + "|" + p.extractor
    v = pd.read_parquet(RES / f"cfv_{ds}_{mo}.parquet")[["key", f"cf2_{sub}"]].rename(columns={f"cf2_{sub}": "CF"})
    d = p.merge(v, on="key")
    return d[(d.extractor != "pls_k1") & (d.family != "ne_lr")]


def fmt(x):
    s = f"{x:.2f}"
    return s.replace("-0.", "$-$.").replace("0.", ".")


def fmt3(x):
    return f"{x:+.3f}".replace("-0.", "$-$.").replace("+0.", "+.")


rows = []
for ds in DSN:
    for mo, lab in [("qwen7b", "mean"), ("qwen7b_last", "last token")]:
        if not (RES / f"cfv_{ds}_{mo}.parquet").exists():
            continue
        d = load(ds, mo)
        mp = d.groupby(["rho", "layer", "extractor"], as_index=False)[["test_dec", "CF"]].mean()
        r = {"data": DSN[ds], "readout": lab}
        for rho in [0.5, 0.9]:
            g = mp[mp.rho == rho]
            n = g[g.test_dec >= g.test_dec.max() - 0.01]
            r[f"best{rho}"] = f"{g.test_dec.max():.3f}"[1:]
            r[f"cf{rho}"] = f"{fmt(n.CF.min())}--{fmt(n.CF.max())}"
        s9 = d[d.rho == 0.9]
        a = s9.loc[s9.groupby("seed").dev_dec.idxmax()]
        b = s9.loc[s9.groupby("seed").dev_conflict.idxmax()]
        r["sel"] = f"{fmt(a.CF.mean())}$\\to${fmt(b.CF.mean())} ({fmt3(b.test_dec.mean() - a.test_dec.mean())})"
        rows.append(r)
lines = [r"\begin{table}[h]", r"\centering\small\setlength{\tabcolsep}{2pt}", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{$\rhot=0.9$} & \multicolumn{2}{c}{$\rhot=0.5$} & selection, $\rhot=0.9$ \\",
         r"data & readout & best \dec{} & $\cfn$ near-best & best \dec{} & $\cfn$ near-best & $\cfn$: \dec{}$\to$\conf{} ($\Delta$\dec{}) \\",
         r"\midrule"]
for r in rows:
    lines.append(f"{r['data']} & {r['readout']} & {r['best0.9']} & {r['cf0.9']} & {r['best0.5']} & {r['cf0.5']} & {r['sel']} \\\\")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Readout ablation (Qwen2.5-7B, standard extractors, accepted edits): mean pooling vs.\ the last non-special token.}",
          r"\label{tab:pooling}", r"\end{table}"]
(ROOT / "paper/tables/tab_pooling.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
