"""Paper figures -> paper/figs/*.pdf
fig1: decodability (x) vs. counterfactual nuisance sensitivity (y), one point per extractor (mean over 3
      training resamples), rows = training regime, columns = dataset; Qwen (Gemma in fig_gemma)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# reference palette slots 1-3 (validated all-pairs) + neutral for the crude baseline; shapes are secondary encoding
STYLE = {
    "lr": dict(c="#2a78d6", marker="o", label="logistic probe"),
    "pls": dict(c="#eb6834", marker="s", label="PLS subspace"),
    "dom": dict(c="#5f5e5a", marker="^", label="diff-of-means"),
    "ne_lr": dict(c="#1baf7a", marker="D", label="nuisance-erased probe"),
}
COMPETITIVE = 0.05
DS = {"cebab": "CEBaB: service | food", "bios": "BiasBios: surgeon | gender", "civil": "CivilComments: toxic | identity"}

pop = pd.concat([pd.read_parquet(p) for p in sorted(RES.glob("pop_*.parquet"))], ignore_index=True)
num = [c for c in pop.columns if c.startswith(("test_", "dev_", "cf_"))]
mp = pop.groupby(["dataset", "model", "rho", "layer", "extractor", "family"], as_index=False)[num].mean()


def scatter(model, ycol, ylabel, fname, rhos=(0.5, 0.9), near=0.01):
    fig, axes = plt.subplots(len(rhos), 3, figsize=(6.3, 1.5 * len(rhos) + 0.35), squeeze=False)
    for i, rho in enumerate(rhos):
        for j, ds in enumerate(DS):
            ax = axes[i, j]
            g = mp[(mp.model == model) & (mp.dataset == ds) & (mp.rho == rho)]
            g = g[g.test_dec >= g.test_dec.max() - COMPETITIVE]  # the claim concerns competitive extractors
            if g.empty:
                ax.set_axis_off()
                continue
            best = g.test_dec.max()
            ax.axvspan(best - near, best + 0.002, color=GRID, lw=0, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
            for fam in ["pls", "lr", "ne_lr", "dom"]:  # dom last: few points, keep them visible
                h = g[g.family == fam]
                st = STYLE[fam]
                ax.scatter(h.test_dec, h[ycol], s=14 if fam == "dom" else 9, c=st["c"], marker=st["marker"], linewidths=0.4,
                           edgecolors="white", label=st["label"], zorder=3, alpha=0.9)
            ax.set_xlim(best - COMPETITIVE - 0.002, best + 0.004)
            ax.grid(axis="y", color=GRID, lw=0.4)
            if i == 0:
                ax.set_title(DS[ds], color=INK)
            if j == 0:
                ax.set_ylabel(f"$\\rho_{{train}}={rho}$\n{ylabel}")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.supxlabel("decodability: in-distribution AUROC (shaded: within 0.01 of the best extractor)", y=0.06, fontsize=7.5)
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), markerscale=1.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.savefig(OUT / fname, bbox_inches="tight")
    fig.savefig(OUT / fname.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


for model in sorted(mp.model.unique()):
    scatter(model, "cf_nuisance", "CF nuisance sensitivity", f"fig1_cf_{model}.pdf")
    scatter(model, "test_conflict", "conflict AUROC", f"fig_conflict_{model}.pdf")
print("figures ->", OUT)
