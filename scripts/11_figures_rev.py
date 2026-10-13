"""Revised figures -> paper/figs/
fig_main_<model>: test decodability (x) vs validated counterfactual nuisance sensitivity CF_N (y) for competitive
  extractors (seed means), rows = training regime, columns = dataset; overlaid: the extractor each of the 3
  training samples selects on DEV data by max decodability (black ring) vs max Conflict (black star), standard pool.
fig_synthetic: CF vs overlap at rho=0.5 with decodability alongside (appendix)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "figs"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"pls": dict(c="#eb6834", marker="s", label="PLS subspace"),
         "lr": dict(c="#2a78d6", marker="o", label="logistic probe"),
         "ne_lr": dict(c="#1baf7a", marker="D", label="nuisance-erased probe"),
         "dom": dict(c="#5f5e5a", marker="^", label="diff-of-means")}
DS = {"cebab": "CEBaB: service | food", "bios": "BiasBios: surgeon | gender", "civil": "Civil: toxic | identity"}
COMP = 0.05

sub = json.load(open(RES / "rev_key_numbers.json"))["cf_pair_subset"]
rows = []
for ds in DS:
    for mo in ["qwen7b", "gemma9b"]:
        p = pd.read_parquet(RES / f"pop_{ds}_{mo}.parquet")
        p["key"] = p.layer.astype(str) + "|" + p.rho.astype(str) + "|" + p.seed.astype(str) + "|" + p.extractor
        v = pd.read_parquet(RES / f"cfv_{ds}_{mo}.parquet")[["key", f"cf2_{sub}"]].rename(columns={f"cf2_{sub}": "CF"})
        rows.append(p.merge(v, on="key"))
pop = pd.concat(rows)
pop = pop[pop.extractor != "pls_k1"]
mp = pop.groupby(["dataset", "model", "rho", "layer", "extractor", "family"], as_index=False)[["test_dec", "CF"]].mean()
sel = pd.read_csv(RES / "rev_D_selection_by_seed.csv")
sel = sel[(sel.pool == "standard") & sel.criterion.isin(["max_dec", "max_conflict"])]


def main_fig(model, rhos=(0.5, 0.9)):
    fig, axes = plt.subplots(len(rhos), 3, figsize=(6.3, 1.5 * len(rhos) + 0.45), squeeze=False)
    for i, rho in enumerate(rhos):
        for j, ds in enumerate(DS):
            ax = axes[i, j]
            g = mp[(mp.model == model) & (mp.dataset == ds) & (mp.rho == rho)]
            best = g.test_dec.max()
            g = g[g.test_dec >= best - COMP]
            ax.axvspan(best - 0.01, best + 0.002, color=GRID, lw=0, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
            for fam in ["pls", "lr", "ne_lr", "dom"]:
                h = g[g.family == fam]
                st = STYLE[fam]
                ax.scatter(h.test_dec, h.CF, s=8, c=st["c"], marker=st["marker"], linewidths=0.3,
                           edgecolors="white", label=st["label"], zorder=3, alpha=0.75)
            s = sel[(sel.model == model) & (sel.dataset == ds) & (sel.rho == rho)]
            a = s[s.criterion == "max_dec"]
            b = s[s.criterion == "max_conflict"]
            ax.scatter(a.test_dec, a.CF, s=34, facecolors="none", edgecolors=INK, linewidths=0.9, zorder=5,
                       label="selected: max dev decodability")
            ax.scatter(b.test_dec, b.CF, s=48, marker="*", c=INK, linewidths=0, zorder=6,
                       label="selected: max dev Conflict")
            ax.set_xlim(best - COMP - 0.002, best + 0.004)
            ax.grid(axis="y", color=GRID, lw=0.4)
            if i == 0:
                ax.set_title(DS[ds], color=INK)
            if j == 0:
                ax.set_ylabel(f"$\\rho_{{tr}}={rho}$\n" + r"$\mathrm{CF}_N$ (validated edits)")
    h, l = axes[-1, 0].get_legend_handles_labels()
    fig.supxlabel("test decodability, in-distribution AUROC (shaded: within 0.01 of the best)", y=0.075, fontsize=7.5)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.025), markerscale=1.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.6, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_main_{model}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


for m in ["qwen7b", "gemma9b"]:
    main_fig(m)

# synthetic (appendix)
R = pd.read_csv(RES / "rev_synthetic.csv")
R = R[R.rho == 0.5].groupby(["overlap", "extractor"], as_index=False)[["cf", "dec"]].mean()
fig, axes = plt.subplots(1, 2, figsize=(3.1, 1.4))
lab = {"dom": ("diff-of-means", STYLE["dom"]), "lr_C0.01": ("logistic probe", STYLE["lr"]),
       "pls_k8": ("PLS-8", STYLE["pls"]), "ne_lr_C0.01": ("nuisance-erased", STYLE["ne_lr"])}
for e, (name, st) in lab.items():
    h = R[R.extractor == e]
    axes[0].plot(h.overlap, h.cf, marker=st["marker"], c=st["c"], lw=1, ms=3, label=name)
    axes[1].plot(h.overlap, h.dec, marker=st["marker"], c=st["c"], lw=1, ms=3)
axes[0].set(xlabel="overlap cos(u, v)", ylabel=r"$\mathrm{CF}_N$")
axes[1].set(xlabel="overlap cos(u, v)", ylabel="decodability", ylim=(0.5, 1.0))
for ax in axes:
    ax.grid(color=GRID, lw=0.4)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2))
fig.tight_layout(rect=(0, 0.12, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_synthetic.{ext}", bbox_inches="tight", dpi=200)
print("ok")
