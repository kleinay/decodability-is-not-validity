"""Generate every LaTeX table and in-text number of the paper from results/*.csv -> paper/tables/*.tex and
paper/tables/numbers.tex (\\newcommand macros). Nothing in the paper is typed by hand."""
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from dv.datasets import ALL, DESC, MAIN, NAME as DSN  # noqa: E402
MN = {"qwen7b": "Q", "gemma9b": "G"}
macros = {}


def f2(x, signed=False):
    if pd.isna(x):
        return "--"
    s = f"{x:.2f}"
    if s.startswith("-0."):
        s = "$-$." + s[3:]
    elif s.startswith("0."):
        s = ("+" if signed and x > 0 else "") + "." + s[2:]
    elif s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def f3(x, signed=True):
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "$-$").replace("+0.", "+.").replace("$-$0.", "$-$.").replace("0.", ".")


def rng_(a, b):
    return f2(a) if f2(a) == f2(b) else f"{f2(a)}--{f2(b)}"


def dec3(x):
    return f"{x:.3f}"[1:] if x < 1 else f"{x:.3f}"


def mac(name, val):
    macros[name] = val


A = pd.read_csv(RES / "rev_A_nearbest.csv")
Bq = pd.read_csv(RES / "rev_B_equivalence.csv")
C = pd.read_csv(RES / "rev_C_correlations.csv")
Dm = pd.read_csv(RES / "rev_D_selection_means.csv")
Dd = pd.read_csv(RES / "rev_D_selection_deltas.csv")
E = pd.read_csv(RES / "rev_E_fixed_pair.csv")
F = pd.read_csv(RES / "rev_F_geometry.csv")
G = pd.read_csv(RES / "rev_G_judges.csv")
Gag = pd.read_csv(RES / "rev_G_agreement.csv") if (RES / "rev_G_agreement.csv").stat().st_size > 5 else pd.DataFrame()
Gcf = pd.read_csv(RES / "rev_G_cf_robustness.csv")
KN = json.load(open(RES / "rev_key_numbers.json"))

# ---------------------------------------------------------------- Table: near-best (main, standard pool)
def near_table(rhos, fname, caption_rho, dss):
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r" & & best & \#near & $\cfn$ near-best & $\cfn$ equiv. & \conf{} \\",
             r" & & \dec{} & (equiv.) & range & range & range \\", r"\midrule"]
    for rho in rhos:
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{caption_rho[rho]}}}}}\\")
        for ds in dss:
            for mo in MN:
                a = A[(A.dataset == ds) & (A.model == mo) & (A.rho == rho) & (A.pool == "standard") & (A.defined_on == "test_dec")].iloc[0]
                b = Bq[(Bq.dataset == ds) & (Bq.model == mo) & (Bq.rho == rho) & (Bq.pool == "standard")]
                eq = rng_(b.CF_min.iloc[0], b.CF_max.iloc[0]) if len(b) and b.n_equiv.iloc[0] > 1 else (f2(b.CF_min.iloc[0]) if len(b) else "--")
                neq = int(b.n_equiv.iloc[0]) if len(b) else 0
                lines.append(f"{DSN[ds] if mo == 'qwen7b' else ''} & {MN[mo]} & {dec3(a.best_test_dec)} & {int(a.n_near)} ({neq}) & "
                             f"{rng_(a.CF_min, a.CF_max)} & {eq} & {rng_(a.test_conflict_min, a.test_conflict_max)} \\\\")
        if rho != rhos[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / fname).write_text("\n".join(lines) + "\n")


near_table([0.9], "tab_near.tex", {0.9: r"$\rhot=0.9$ (correlated training data)"}, MAIN)
near_table([0.9, 0.5], "tab_near_all.tex", {0.9: r"$\rhot=0.9$ (correlated training data)", 0.5: r"$\rhot=0.5$ (decorrelated)"}, ALL)
near_table([0.7], "tab_near_rho07.tex", {0.7: r"$\rhot=0.7$"}, ALL)

# ---------------------------------------------------------------- Table: selection
crit = ["max_dec", "max_conflict", "min_leak_at_dec", "max_dec_minus_leak"]
lines = [r"\begin{tabular}{@{}ll|cccc|ccc@{}}", r"\toprule",
         r" & & \multicolumn{4}{c|}{standard extractors: $\cfn$ ($\Delta$\dec{})} & \multicolumn{3}{c}{+ erased probe: $\cfn$ ($\Delta$\dec{})} \\",
         r" & & max \dec{} & max \conf{} & min $|$\leak{}$|$ & \dec{}$-|$\leak{}$|$ & max \conf{} & min $|$\leak{}$|$ & \dec{}$-|$\leak{}$|$ \\",
         r"\midrule"]
for ds in DSN:
    for mo in MN:
        cells = []
        for pool, cs in [("standard", crit), ("full", crit[1:])]:
            for c in cs:
                m = Dm[(Dm.dataset == ds) & (Dm.model == mo) & (Dm.rho == 0.9) & (Dm.pool == pool) & (Dm.criterion == c)].iloc[0]
                if c == "max_dec":
                    cells.append(f2(m.CF))
                else:
                    d = Dd[(Dd.dataset == ds) & (Dd.model == mo) & (Dd.rho == 0.9) & (Dd.pool == pool) & (Dd.criterion == c)].iloc[0]
                    mark = "" if d.d_CF_worst < 0 else r"$^\dagger$"
                    cells.append(f"{f2(m.CF)}{mark} ({f3(d.d_dec)})")
        lines.append(f"{DSN[ds] if mo == 'qwen7b' else ''} & {MN[mo]} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_select.tex").write_text("\n".join(lines) + "\n")

# selection macros (rho = 0.9)
sel = Dd[(Dd.rho == 0.9) & Dd.dataset.isin(MAIN)]  # main-text claims pool the main datasets only
s_std_conf = sel[(sel.pool == "standard") & (sel.criterion == "max_conflict")]
s_full_conf = sel[(sel.pool == "full") & (sel.criterion == "max_conflict")]
mac("selStdConfAllSeeds", str(bool((s_std_conf.d_CF_worst < 0).all())))
mac("selStdConfCFmin", f2(-s_std_conf.d_CF.max()))
mac("selStdConfCFmax", f2(-s_std_conf.d_CF.min()))
sens = s_std_conf[s_std_conf.dataset.isin(["cebab", "bios", "amzfp"])]   # settings where the default pick is sensitive
mac("selSensConfCFmin", f2(-sens.d_CF.max()))
mac("selSensConfCFmax", f2(-sens.d_CF.min()))
mac("selSensConfDecMeanWorst", f3(-sens.d_dec.min(), signed=False))
mac("selSensConfDecWorst", f3(-sens.d_dec_worst.min(), signed=False))
mac("selSensNfail", str(int((sens.d_CF_worst >= 0).sum())))
mac("selFullConfAllSeeds", str(bool((s_full_conf.d_CF_worst < 0).all())))
mac("selFullConfDecWorstSample", f3(-s_full_conf.d_dec_worst.min(), signed=False))
mac("selStdConfDecWorst", f3(s_std_conf.d_dec_worst.min(), signed=False).replace("$-$", ""))
mac("selFullConfCFmin", f2(-s_full_conf.d_CF.max()))
mac("selFullConfCFmax", f2(-s_full_conf.d_CF.min()))
mac("selFullConfDecWorst", f3(-s_full_conf.d_dec_worst.min(), signed=False))
mac("selStdConfDecMeanWorst", f3(-s_std_conf.d_dec.min(), signed=False))
mac("selFullConfDecMeanWorst", f3(-s_full_conf.d_dec.min(), signed=False))
selD = pd.read_csv(RES / "rev_D_selection_by_seed.csv")
selF = selD[(selD.rho == 0.9) & (selD.pool == "full") & selD.dataset.isin(MAIN)]
share = selF.groupby("criterion").extractor.apply(lambda x: x.str.startswith("ne_").mean())
mac("neShareMinLeak", f"{share['min_leak_at_dec']:.0%}".replace("%", "\\%"))
nfail = sel[sel.d_CF_worst >= 0][["dataset", "model", "pool", "criterion"]]
mac("selNfailCells", str(len(nfail)))
mac("selNcells", str(len(sel)))

# ---------------------------------------------------------------- Table: datasets with edit validation
n_generated = {ds: pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet").groupby("kind").size().to_dict() for ds in DSN}
rows = []
for ds in DSN:
    g = G[G.dataset == ds]
    acc = {}
    for kind in g.kind.unique():
        if not Gag.empty and ((Gag.dataset == ds) & (Gag.kind == kind)).any():
            acc[kind] = Gag[(Gag.dataset == ds) & (Gag.kind == kind)].accept_both.iloc[0]
        else:
            acc[kind] = g[g.kind == kind].accepted.mean()
    rows.append((ds, acc, g.groupby("kind").n.max().to_dict()))
lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r" & target $T$ / nuisance $N$ & $N$ edits & judged by both & accepted \\", r"\midrule"]
desc = DESC
for ds in DSN:
    kinds = ["nuisance", "nuisance_rev"] if ds == "civil" else ["nuisance"]
    gen = "+".join(str(int(n_generated[ds][k])) for k in kinds)
    jb = "+".join(str(int(Gag[(Gag.dataset == ds) & (Gag.kind == k)].n.iloc[0])) for k in kinds)
    acc = ", ".join(f"{100 * Gag[(Gag.dataset == ds) & (Gag.kind == k)].accept_both.iloc[0]:.0f}\\%" for k in kinds)
    lines.append(f"{DSN[ds] + ('$^*$' if ds not in MAIN else '')} & {desc[ds]} & {gen} & {jb} & {acc} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_data.tex").write_text("\n".join(lines) + "\n")

# judges appendix table
lines = [r"\begin{tabular}{@{}llrcccccc@{}}", r"\toprule",
         r"data & edit kind & $n$ & judge & $T$ kept & $N$ changed & fluency drop & major other & accepted \\", r"\midrule"]
for _, r in G.sort_values(["dataset", "kind", "judge"]).iterrows():
    lines.append(f"{DSN[r.dataset]} & {r.kind.replace('_', ' ')} & {r.n} & {r.judge.replace('_', '-')} & {f2(r.target_same)} & "
                 f"{f2(r.nuis_changed)} & {f2(r.fluency_drop)} & {f2(r.other_major)} & {f2(r.accepted)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_judges.tex").write_text("\n".join(lines) + "\n")
if not Gag.empty:
    lines = [r"\begin{tabular}{@{}llrcccc@{}}", r"\toprule",
             r"data & edit kind & $n$ & accept (Qwen3) & accept (gpt-oss) & both & $\kappa$ \\", r"\midrule"]
    for _, r in Gag.iterrows():
        lines.append(f"{DSN[r.dataset]} & {r.kind.replace('_', ' ')} & {r.n} & {f2(r.accept_qwen)} & {f2(r.accept_gptoss)} & "
                     f"{f2(r.accept_both)} & {f2(r.kappa)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_agreement.tex").write_text("\n".join(lines) + "\n")
    mac("kappaMin", f2(Gag.kappa.min()))
    mac("kappaMax", f2(Gag.kappa.max()))

# ---------------------------------------------------------------- correlations appendix table
lines = [r"\begin{tabular}{@{}lllcccc@{}}", r"\toprule",
         r"data & model & $\rhot$ & $\cfn$ & $|\cfn|$ & \conf{} & $|$\leak{}$|$ \\", r"\midrule"]
for _, r in C[C.pool == "standard"].iterrows():
    cell = lambda c: f"{f2(r[f'rho_{c}'], True)} [{f2(r[f'rho_{c}_lo'], True)}, {f2(r[f'rho_{c}_hi'], True)}]"  # noqa: E731
    lines.append(f"{DSN[r.dataset]} & {MN[r.model]} & {r.rho} & {cell('CF')} & {cell('CFABS')} & {cell('test_conflict')} & {cell('test_leak_abs')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_corr.tex").write_text("\n".join(lines) + "\n")
c9 = C[(C.pool == "standard") & (C.rho == 0.9) & C.dataset.isin(MAIN)]
mac("corrCFmin", f2(c9.rho_CF.min(), True))
mac("corrCFmax", f2(c9.rho_CF.max(), True))
mac("corrCFnSigNeg", str(int((c9.rho_CF_hi < 0).sum())))
mac("corrCFnSigPos", str(int((c9.rho_CF_lo > 0).sum())))
mac("corrCFnNS", str(int(((c9.rho_CF_lo <= 0) & (c9.rho_CF_hi >= 0)).sum())))

# ---------------------------------------------------------------- robustness appendix table
cols = [("spearman_all_vs_val", "all vs.\\ accepted"), ("spearman_cf_vs_cfsd", "$\\Delta_T$ vs.\\ SD norm."),
        ("spearman_cf_vs_cfabs", "signed vs.\\ $|\\cdot|$"), ("spearman_dirpos_vs_dirneg", "direction $+$ vs.\\ $-$"),
        ("spearman_fwd_vs_rev", "Civil fwd vs.\\ rev")]
lines = [r"\begin{tabular}{@{}lll" + "c" * len(cols) + r"@{}}", r"\toprule",
         "data & model & $\\rhot$ & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
for _, r in Gcf.iterrows():
    lines.append(f"{DSN[r.dataset]} & {MN[r.model]} & {r.rho} & " + " & ".join(f2(r.get(c[0]), True) for c in cols) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_robust.tex").write_text("\n".join(lines) + "\n")

# placebo floors and signed-vs-abs at rho = 0.5 (competitive medians)
for ds in DSN:
    g = Gcf[(Gcf.dataset == ds)]
    for rho in [0.5, 0.9]:
        r = g[g.rho == rho]
        for mo in MN:
            rr = r[r.model == mo].iloc[0]
            k = f"{ds}{MN[mo]}{'Five' if rho == 0.5 else 'Nine'}"
            mac(f"absCF{k}", f2(rr.get("cfabs2_val_median", rr.get("cfabs2_all_median"))))
            mac(f"sgnCF{k}", f2(rr.get("cf2_val_median", rr.get("cf2_all_median"))))
            pl = rr.get("cfabs_placebo_val_median", rr.get("cfabs_placebo_all_median", np.nan))
            mac(f"placebo{k}", f2(pl))

# ---------------------------------------------------------------- fixed pair
for _, r in E.iterrows():
    k = f"{r.dataset}{MN[r.model]}{'Five' if r.rho == 0.5 else 'Nine'}"
    mac(f"fpDd{k}", f3(r.d_dec))
    mac(f"fpLo{k}", f3(r.d_dec_ci90_lo))
    mac(f"fpHi{k}", f3(r.d_dec_ci90_hi))
    mac(f"fpEq{k}", "yes" if r.equivalent_0p01 else "no")
    mac(f"fpCFlr{k}", f2(r.CF_lr))
    mac(f"fpCFne{k}", f2(r.CF_ne))
lines = [r"\begin{tabular}{@{}lllcccc@{}}", r"\toprule",
         r"data & model & $\rhot$ & $\Delta$\dec{} [90\% CI] & equiv. & $\cfn$ probe & $\cfn$ erased \\", r"\midrule"]
for _, r in E.iterrows():
    lines.append(f"{DSN[r.dataset]} & {MN[r.model]} & {r.rho} & {f3(r.d_dec)} [{f3(r.d_dec_ci90_lo)}, {f3(r.d_dec_ci90_hi)}] & "
                 f"{'yes' if r.equivalent_0p01 else 'no'} & {f2(r.CF_lr)} & {f2(r.CF_ne)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_fixed.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- geometry
for ds in DSN:
    g = F[F.dataset == ds]
    mac(f"cosMin{ds}", f2(g.cos_T_N.min()))
    mac(f"cosMax{ds}", f2(g.cos_T_N.max()))
    mac(f"predMin{ds}", f2(g.dom_cf_predicted.min()))
    mac(f"predMax{ds}", f2(g.dom_cf_predicted.max()))
    mac(f"obsMin{ds}", f2(g.dom_cf_observed.min()))
    mac(f"obsMax{ds}", f2(g.dom_cf_observed.max()))
for ds in ["cebab", "bios"]:  # within-dataset, across layers and models (pooling datasets would only reflect two clusters)
    g = F[F.dataset == ds]
    mac(f"geomPearson{ds}", f2(np.corrcoef(g.dom_cf_predicted, g.dom_cf_observed)[0, 1]))
    mac(f"geomMAE{ds}", f2((g.dom_cf_predicted - g.dom_cf_observed).abs().mean()))

# ---------------------------------------------------------------- near-best macros (rho 0.9 / 0.5, standard)
for _, a in A[(A.pool == "standard") & (A.defined_on == "test_dec")].iterrows():
    k = f"{a.dataset}{MN[a.model]}{ {0.5: 'Five', 0.7: 'Seven', 0.9: 'Nine'}[a.rho] }".replace(" ", "")
    mac(f"nbCFmin{k}", f2(a.CF_min))
    mac(f"nbCFmax{k}", f2(a.CF_max))
    mac(f"nbConfMin{k}", f2(a.test_conflict_min))
    mac(f"nbConfMax{k}", f2(a.test_conflict_max))
    mac(f"nbN{k}", str(int(a.n_near)))
    mac(f"nbBest{k}", dec3(a.best_test_dec))
for _, b in Bq[Bq.pool == "standard"].iterrows():
    k = f"{b.dataset}{MN[b.model]}{'Five' if b.rho == 0.5 else 'Nine'}"
    mac(f"eqN{k}", str(int(b.n_equiv)))
    mac(f"eqCFmin{k}", f2(b.CF_min))
    mac(f"eqCFmax{k}", f2(b.CF_max))

# ---------------------------------------------------------------- CEBaB target edits (convergent) vs nuisance edits, near-best
sub = KN["cf_pair_subset"]
cf_t, cf_n = [], []
for mo in MN:
    p = pd.read_parquet(RES / f"pop_cebab_{mo}.parquet")
    p["key"] = p.layer.astype(str) + "|" + p.rho.astype(str) + "|" + p.seed.astype(str) + "|" + p.extractor
    v = pd.read_parquet(RES / f"cfv_cebab_{mo}.parquet")[["key", f"cf_target_{sub}", f"cf2_{sub}"]]
    d = p.merge(v, on="key")
    d = d[d.extractor != "pls_k1"]
    mp_ = d.groupby(["rho", "layer", "extractor"], as_index=False)[["test_dec", f"cf_target_{sub}", f"cf2_{sub}"]].mean()
    for _, g in mp_.groupby("rho"):
        n = g[g.test_dec >= g.test_dec.max() - 0.01]
        cf_t += list(n[f"cf_target_{sub}"]); cf_n += list(n[f"cf2_{sub}"])
mac("cfTmin", f2(min(cf_t)))
mac("cfTmax", f2(max(cf_t)))
mac("cfNnbMin", f2(min(cf_n)))
mac("cfNnbMax", f2(max(cf_n)))

# ---------------------------------------------------------------- robustness / placebo macros
Gm = Gcf[Gcf.dataset.isin(MAIN)]
mac("robAllValMin", f2(Gm.spearman_all_vs_val.min()))
mac("robSDMin", f2(Gm.spearman_cf_vs_cfsd.min()))
mac("robAbsMinSevenUp", f2(Gm[Gm.rho >= 0.7].spearman_cf_vs_cfabs.min()))
mac("robAbsMinNine", f2(Gm[Gm.rho == 0.9].spearman_cf_vs_cfabs.min()))
mac("robDirMinNine", f2(Gm[Gm.rho == 0.9].spearman_dirpos_vs_dirneg.min()))
pcol = "cfabs_placebo_val_median" if "cfabs_placebo_val_median" in Gcf else "cfabs_placebo_all_median"
mac("placeboMax", f2(Gm[pcol].max()))
mac("placeboMin", f2(Gm[pcol].min()))
g9 = Gm[Gm.rho == 0.9]
acol = "cfabs2_val_median" if "cfabs2_val_median" in Gcf else "cfabs2_all_median"
mac("nuisAbsNineMin", f2(g9[acol].min()))
mac("nuisAbsNineMax", f2(g9[acol].max()))
if not Gag.empty:
    pl = Gag[Gag.kind == "placebo"]
    mac("qwenPlaceboAccMin", f2(pl.accept_qwen.min()))
    mac("qwenPlaceboAccMax", f2(pl.accept_qwen.max()))
    mac("gptPlaceboAccMin", f2(pl.accept_gptoss.min()))
    mac("gptPlaceboAccMax", f2(pl.accept_gptoss.max()))

Syn = pd.read_csv(RES / "rev_synthetic.csv")
sy = Syn[Syn.rho == 0.5].groupby(["extractor", "overlap"]).mean(numeric_only=True).reset_index()
rngs = sy.groupby("extractor").agg(dmin=("dec", "min"), dmax=("dec", "max"), cmin=("cf", "min"), cmax=("cf", "max"))
std_ = rngs.drop(index="ne_lr_C0.01")
mac("synDecRangeStd", f2((std_.dmax - std_.dmin).max()))
mac("synDecRangeNe", f2(rngs.loc["ne_lr_C0.01"].dmax - rngs.loc["ne_lr_C0.01"].dmin))
mac("synCFRangeStdMin", f2((std_.cmax - std_.cmin).min()))
mac("synCFRangeStdMax", f2((std_.cmax - std_.cmin).max()))

for ds in ["amzneg", "amzfp"]:
    for rho, rn in [(0.5, "Five"), (0.9, "Nine")]:
        g = Gcf[(Gcf.dataset == ds) & (Gcf.rho == rho)]
        mac(f"dirPos{ds}{rn}", rng_(g.cf_nuisance_dirpos_median.min(), g.cf_nuisance_dirpos_median.max()))
        mac(f"dirNeg{ds}{rn}", rng_(g.cf_nuisance_dirneg_median.min(), g.cf_nuisance_dirneg_median.max()))
        mac(f"sgn{ds}{rn}", rng_(g.cf2_val_median.min(), g.cf2_val_median.max()))
        mac(f"abs{ds}{rn}", rng_(g.cfabs2_val_median.min(), g.cfabs2_val_median.max()))
        mac(f"plc{ds}{rn}", rng_(g.cfabs_placebo_val_median.min(), g.cfabs_placebo_val_median.max()))
cg = Gag[Gag.dataset == "civil"].set_index("kind").accept_both
mac("civilAccFwd", f"{100 * cg['nuisance']:.0f}\\%"); mac("civilAccRev", f"{100 * cg['nuisance_rev']:.0f}\\%")
for ds in ["amzneg", "amzfp"]:
    g = Gcf[Gcf.dataset == ds].pivot_table(index="model", columns="rho", values="cf2_val_median")
    inc = g[0.9] - g[0.5]
    mac(f"inc{ds}", rng_(inc.min(), inc.max()))
nat = json.load(open(ROOT / "data/amz/natural_rates.json"))
mac("amzNegNeg", f2(nat["P(NEG|negative)"])); mac("amzNegPos", f2(nat["P(NEG|positive)"]))
mac("amzFpNeg", f2(nat["P(FP|negative)"])); mac("amzFpPos", f2(nat["P(FP|positive)"]))

# ---------------------------------------------------------------- judge validation (14_judge_validation.py, 15_filter_robustness.py)
if (RES / "rev_J_cebab_gold.csv").exists():
    J = pd.read_csv(RES / "rev_J_cebab_gold.csv").set_index(["judge", "question"])
    qs = ["target_same", "nuis_same"]
    mac("jvGptQ", rng_(*sorted(J.loc[[("j2", q) for q in qs], "acc"])))
    mac("jvQwenQ", rng_(*sorted(J.loc[[("j1", q) for q in qs], "acc"])))
    mac("jvHumanQ", rng_(*sorted(J.loc[[("j1", q) for q in qs], "human_single_vs_majority"])))
    e = J.loc[("both", "accept_edits")]
    mac("jvN", str(int(e.n))); mac("jvNbad", str(int(e.n_gold_bad)))
    mac("jvBadRate", f2(e.n_gold_bad / e.n)); mac("jvGoodRate", f2(1 - e.n_gold_bad / e.n)); mac("jvPrec", f2(e.precision))
    mac("jvRejBad", f2(e.reject_gold_bad)); mac("jvRecall", f2(e.recall_gold_ok))
    Pl = pd.read_csv(RES / "rev_J_planted.csv")
    PT = {"clean": "none (retest)", "flip": "opinion reversed", "strength": "opinion stronger/weaker",
          "occupation": "occupation changed", "mixed_gender": "gender cues mixed", "garble": "garbled (rule)",
          "fact": "minor fact changed"}
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule", r"data & planted error & Qwen & gpt-oss & both \\", r"\midrule"]
    for ds in ["amzneg", "amzfp", "bios"]:
        for i, pt in enumerate([p for p in PT if p in set(Pl[Pl.dataset == ds].ptype)]):
            r = Pl[(Pl.dataset == ds) & (Pl.ptype == pt)].iloc[0]
            lines.append(f"{DSN[ds] if i == 0 else ''} & {PT[pt]} & {f2(r.j1_accept)} & {f2(r.j2_accept)} & {f2(r.both_accept)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_planted.tex").write_text("\n".join(lines) + "\n")
    bad = Pl[Pl.ptype.isin(["flip", "occupation", "garble"])]
    mac("plBadMax", f2(bad.both_accept.max())); mac("plFlipMax", f2(Pl[Pl.ptype == "flip"].both_accept.max()))
    mac("plStrGptMin", f2(Pl[Pl.ptype == "strength"].j2_accept.min())); mac("plStrGptMax", f2(Pl[Pl.ptype == "strength"].j2_accept.max()))
    mac("plStrBothMax", f2(Pl[Pl.ptype == "strength"].both_accept.max()))
    mac("plMixBoth", f2(Pl[Pl.ptype == "mixed_gender"].both_accept.max()))
    cl = Pl[Pl.ptype == "clean"].both_accept
    mac("plCleanMin", f2(cl.min())); mac("plCleanMax", f2(cl.max()))
    fa = Pl[Pl.ptype == "fact"].both_accept
    mac("plFactMin", f2(fa.min())); mac("plFactMax", f2(fa.max()))
if (RES / "rev_I_filter_robustness.csv").exists():
    I = pd.read_csv(RES / "rev_I_filter_robustness.csv")
    Im = I[I.dataset.isin(MAIN) & I.rho.isin([0.5, 0.9])]
    mac("frSpearMin", f2(Im.spearman_vs_both.min()))
    I9 = Im[Im.rho == 0.9]
    mac("frSelAllLower", str(bool((I9.selconf_dCF_full < 0).all())))
    mac("frSelFullMin", f2(-I9.selconf_dCF_full.max())); mac("frSelFullMax", f2(-I9.selconf_dCF_full.min()))
    for f in ["all", "both"]:
        g = I9[(I9.dataset == "cebab")]
        g = g[g["filter"] == f]
        mac(f"frCebab{f.capitalize()}", rng_(g.near_min.min(), g.near_max.max()))
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & all & Qwen & gpt-oss & either & both \\", r"\midrule"]
    for ds in MAIN:
        g = I9[I9.dataset == ds].set_index(["filter", "model"])
        row = []
        for f in ["all", "j1", "j2", "either", "both"]:
            row.append(f"{int(g.loc[(f, 'qwen7b')].n_pairs)}: " + "/".join(
                rng_(g.loc[(f, m)].near_min, g.loc[(f, m)].near_max) for m in ["qwen7b", "gemma9b"]))
        lines.append(f"{DSN[ds]} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_filter.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- LLM readers of accepted Amazon edits (16_author_read_sample.py)
AR = ROOT / "data/author_read"
if (AR / "labels_claude.csv").exists() and (AR / "labels_chatgpt.csv").exists():
    from sklearn.metrics import cohen_kappa_score
    K = pd.read_csv(AR / "key.csv")
    for w in ["claude", "chatgpt"]:
        K[w] = K["item"].map(pd.read_csv(AR / f"labels_{w}.csv").set_index("item").label)
    for ds in ["amzneg", "amzfp"]:
        g = K[K.dataset == ds]
        w = [(g[r] == "wrong").mean() for r in ["claude", "chatgpt"]]
        u = [(g[r] != "ok").mean() for r in ["claude", "chatgpt"]]
        tag = {"amzneg": "Neg", "amzfp": "Fp"}[ds]
        mac(f"rdWrong{tag}", rng_(min(w), max(w)))
        mac(f"rdNotOk{tag}", rng_(min(u), max(u)))
    mac("rdAgree", f2((K.claude == K.chatgpt).mean())); mac("rdKappa", f2(cohen_kappa_score(K.claude, K.chatgpt)))
    sub = K[(K.dataset == "amzneg") & (K.n_src == 0)]
    mac("rdWrongNegAdd", rng_(*sorted((sub[r] == "wrong").mean() for r in ["claude", "chatgpt"])))

with open(OUT / "numbers.tex", "w") as fh:
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and tables:", sorted(p.name for p in OUT.glob("*.tex")))

# ---------------------------------------------------------------- compact selection table for the main text
lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{standard extractors} & + erased \\",
         r" & & max \dec{} & max \conf{} & min $|$\leak{}$|$ & max \conf{} \\", r"\midrule"]
for ds in MAIN:
    for mo in MN:
        cells = []
        for pool, c in [("standard", "max_dec"), ("standard", "max_conflict"), ("standard", "min_leak_at_dec"), ("full", "max_conflict")]:
            m = Dm[(Dm.dataset == ds) & (Dm.model == mo) & (Dm.rho == 0.9) & (Dm.pool == pool) & (Dm.criterion == c)].iloc[0]
            if c == "max_dec":
                cells.append(f2(m.CF))
            else:
                d = Dd[(Dd.dataset == ds) & (Dd.model == mo) & (Dd.rho == 0.9) & (Dd.pool == pool) & (Dd.criterion == c)].iloc[0]
                cells.append(f"{f2(m.CF)}{'' if d.d_CF_worst < 0 else '$^\\dagger$'} ({f3(d.d_dec)})")
        lines.append(f"{DSN[ds] if mo == 'qwen7b' else ''} & {MN[mo]} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_select_main.tex").write_text("\n".join(lines) + "\n")
print("compact selection table written")
