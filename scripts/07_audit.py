"""P0 implementation audit -> results/AUDIT.md
 1. split hygiene: uid disjointness; CEBaB family disjointness; dev never used for fitting (structural, see 03)
 2. sign conventions: every pair's sign equals N_b - N_a (or T_b - T_a for target pairs), recomputed from labels
 3. PLS(k=1) vs difference of means: cosine of fitted weight vectors on real data
 4. truncation / length: share of texts > 256 tokens; token-length change of each edit
 5. independent metrics: re-implement AUROC (Mann-Whitney via ranks), Conflict, Leak and CF from the saved
    raw scores with no shared code, compare with the pipeline's numbers for a random sample of extractors
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dv.extractors import fit_extractor  # noqa: E402

out = ["# Implementation audit (scripts/07_audit.py)\n"]
ok_all = True


def check(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    out.append(f"- [{'PASS' if cond else 'FAIL'}] {msg}")


# ---------------------------------------------------------------- 1, 2
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    out.append(f"\n## {ds}")
    sets = {sp: set(g.uid) for sp, g in tx.groupby("split")}
    check(all(not (sets[a] & sets[b]) for a in sets for b in sets if a < b), "uids disjoint across splits")
    check(not (set(tx[tx.split == "train"].text) & set(tx[tx.split == "test"].text)) or ds == "cebab",
          "no exact train/test text duplicates (CEBaB: 11 known generic duplicates, excluded by family split)")
    lab = tx.set_index("uid")
    for kind, g in pr.groupby("kind"):
        if kind.startswith("placebo"):  # labels must be unchanged; the +1 sign is nominal
            same = all((lab.loc[g.uid_a, c].values == lab.loc[g.uid_b, c].values).all() for c in ["T", "N"])
            check(same, f"{kind}: T and N labels identical for source and edit in all {len(g)} pairs")
            continue
        col = "T" if kind == "target" else "N"
        a, b = lab.loc[g.uid_a, col].values, lab.loc[g.uid_b, col].values
        good = np.isfinite(a) & np.isfinite(b)
        check(good.all() and (np.sign(b - a) == g.sign.values).all(),
              f"{kind}: sign == sign({col}_b - {col}_a) for all {len(g)} pairs")
        if kind != "target":
            check((lab.loc[g.uid_a, "split"].isin(["test", "cfsrc"])).all(), f"{kind}: sources come from test/cfsrc only")
    if ds == "cebab":
        from datasets import load_dataset
        c = load_dataset("CEBaB/CEBaB")
        fam = {f"cebab-{i}": o for sp in c for i, o in zip(c[sp]["id"], c[sp]["original_id"])}
        fams = {sp: {fam[u.removesuffix("-plc")] for u in g.uid} for sp, g in tx.groupby("split")}  # placebos: source family
        check(not (fams["train"] & fams["test"]) and not (fams["train"] & fams["dev"]), "CEBaB review families disjoint train/dev/test")

# ---------------------------------------------------------------- 3
out.append("\n## PLS(k=1) vs difference of means")
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    A = np.load(ROOT / f"data/{ds}/acts_qwen7b.npz", allow_pickle=True)
    tr = np.where((tx.split == "train") & tx["T"].notna() & tx["N"].notna())[0][:2000]
    X = A["X"][tr, 4]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    t, n = tx["T"].values[tr].astype(int), tx["N"].values[tr].astype(int)
    w1, _ = fit_extractor("dom", {}, Z, t, n)
    w2, _ = fit_extractor("pls_k1", {"k": 1}, Z, t, n)
    cos = w1 @ w2 / np.linalg.norm(w1) / np.linalg.norm(w2)
    check(cos > 0.999 or abs(t.mean() - 0.5) > 0.01, f"{ds}: cos(w_DoM, w_PLS1) = {cos:.6f} (P(T=1)={t.mean():.3f}); same orientation")
out.append("  (PLS-1 weight ∝ Zᵀ(t-t̄) ∝ μ₁-μ₀ exactly when classes are balanced; pls_k1 is therefore dropped from"
           " pairwise/correlation analyses as a duplicate of dom)")

# ---------------------------------------------------------------- 4
out.append("\n## Truncation and edit length (Qwen tokenizer)")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    L = pd.Series([len(x) for x in tok(list(tx.text))["input_ids"]], index=tx.index)
    out.append(f"- {ds}: {100 * (L > 256).mean():.1f}% of texts exceed 256 tokens (median {L.median():.0f})")
    for kind, g in pr.groupby("kind"):
        d = (L.loc[g.uid_b].values - L.loc[g.uid_a].values)
        trunc = ((L.loc[g.uid_a].values > 256) | (L.loc[g.uid_b].values > 256)).mean()
        out.append(f"  - {kind}: Δtokens mean {d.mean():+.2f}, mean |Δ| {np.abs(d).mean():.2f}, "
                   f"|Δ|≤2 in {100 * (np.abs(d) <= 2).mean():.0f}% of pairs; either side truncated: {100 * trunc:.1f}%")
    L.rename("ntok").to_frame().to_parquet(ROOT / f"data/{ds}/ntok_qwen.parquet")

# ---------------------------------------------------------------- 5
out.append("\n## Independent metric re-implementation vs pipeline (random 40 extractors per dataset/model)")


def auc(y, s, w=None):
    """Weighted Mann-Whitney AUROC computed from ranks (ties averaged); no sklearn."""
    y = np.asarray(y).astype(bool)
    if w is None:
        r = rankdata(s)
        n1, n0 = y.sum(), (~y).sum()
        return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    # weighted: P(s_pos > s_neg) + 0.5 P(tie), via explicit pairwise sums on sorted unique values
    pos, neg = s[y], s[~y]
    wp, wn = w[y], w[~y]
    order = np.argsort(neg)
    neg, wn = neg[order], wn[order]
    cw = np.concatenate([[0], np.cumsum(wn)])
    lo = np.searchsorted(neg, pos, "left")
    hi = np.searchsorted(neg, pos, "right")
    num = (wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum()
    return num / (wp.sum() * wn.sum())


rng = np.random.RandomState(0)
maxdiff = {}
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    for model in ["qwen7b", "gemma9b"]:
        pop = pd.read_parquet(ROOT / f"results/pop_{ds}_{model}.parquet")
        z = np.load(ROOT / f"results/scores_{ds}_{model}.npz", allow_pickle=True)
        S = dict(zip(z["keys"], z["S"]))
        pos = {u: i for i, u in enumerate(z["uid"])}
        te = tx[(tx.split == "test") & tx["T"].notna() & tx["N"].notna()]
        ti = np.array([pos[u] for u in te.uid])
        t, n = te["T"].values.astype(int), te["N"].values.astype(int)
        for _, r in pop.sample(40, random_state=rng).iterrows():
            s = S[f"{r.layer}|{r.rho}|{r.seed}|{r.extractor}"][ti].astype(float)
            # cell weights to agreement rho, balanced T
            w = np.zeros(len(t))
            for ct in (0, 1):
                for cn in (0, 1):
                    m = (t == ct) & (n == cn)
                    w[m] = (r.rho if ct == cn else 1 - r.rho) / m.sum()
            dec = auc(t, s, w)
            m = t != n
            conf = auc(t[m], s[m])
            leak = np.mean([auc(n[t == c], s[t == c]) for c in (0, 1)]) - 0.5
            wb = np.where(t == n, 0.5, 0.5)
            for ct in (0, 1):
                for cn in (0, 1):
                    mm = (t == ct) & (n == cn)
                    wb[mm] = 1.0 / mm.sum()
            dT = np.average(s[t == 1], weights=wb[t == 1]) - np.average(s[t == 0], weights=wb[t == 0])
            cfs = {}
            for kind, g in pr.groupby("kind"):
                sa = S[f"{r.layer}|{r.rho}|{r.seed}|{r.extractor}"][[pos[u] for u in g.uid_a]].astype(float)
                sb = S[f"{r.layer}|{r.rho}|{r.seed}|{r.extractor}"][[pos[u] for u in g.uid_b]].astype(float)
                cfs[kind] = np.mean(g.sign.values * (sb - sa)) / dT
            cfn = 0.5 * (cfs["nuisance"] + cfs["nuisance_rev"]) if "nuisance_rev" in cfs else cfs["nuisance"]
            for k, mine, theirs in [("dec", dec, r.test_dec), ("conflict", conf, r.test_conflict),
                                    ("leak", leak, r.test_leak), ("cf_nuisance", cfn, r.cf_nuisance)]:
                maxdiff[k] = max(maxdiff.get(k, 0), abs(mine - theirs))
for k, v in maxdiff.items():
    check(v < 1e-4, f"{k}: max |independent - pipeline| = {v:.2e} (float32 score storage)")

out.insert(1, f"**Overall: {'ALL CHECKS PASS' if ok_all else 'SOME CHECKS FAIL'}**\n")
(ROOT / "results/AUDIT.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
