"""Fit the full extractor population for one (dataset, model) and score it with the validity battery.

population = 7 layers x 16 extractor configs x 3 training regimes (rho in {0.5, 0.7, 0.9}) x 3 training resamples
Usage: python scripts/03_fit_population.py <dataset> <model_key>
Writes results/pop_<ds>_<model>.parquet (one row per fitted extractor) and
       results/scores_<ds>_<model>.npz (test/cf scores, for bootstrapping)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dv import validity as V  # noqa: E402
from dv.extractors import configs, family, fit_extractor  # noqa: E402

RHOS = [0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
N_TRAIN = 2000


def sample_train(t, n, rho, seed):
    rng = np.random.RandomState(1000 + seed)
    half = N_TRAIN // 2
    want = {(1, 1): round(rho * half), (1, 0): half - round(rho * half),
            (0, 0): round(rho * half), (0, 1): half - round(rho * half)}
    idx = []
    for (ct, cn), k in want.items():
        pool = np.where((t == ct) & (n == cn))[0]
        assert len(pool) >= k, (ct, cn, len(pool), k)
        idx.append(rng.choice(pool, k, replace=False))
    return np.concatenate(idx)


def run_task(X, li, layer, rho, seed, tr_t, tr_n, tr_pool, ev, pairs):
    tr = tr_pool[sample_train(tr_t, tr_n, rho, seed)]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X[tr] - mu) / sd
    t, n = tr_t_lookup(tr, ev), tr_n_lookup(tr, ev)
    Zall = (X - mu) / sd
    rows, scores = [], {}
    for name, params in configs():
        w, b = fit_extractor(name, params, Z, t, n)
        s = Zall @ w + b
        r = {"layer": layer, "layer_idx": li, "rho": rho, "seed": seed, "extractor": name, "family": family(name)}
        for part in ("test", "dev"):
            m, tt, nn = ev[part]
            r[f"{part}_dec"] = V.dec_auroc(s[m], tt, nn, rho)
            r[f"{part}_dec_bal"] = V.dec_auroc(s[m], tt, nn, 0.5)
            r[f"{part}_conflict"] = V.conflict_auroc(s[m], tt, nn)
            r[f"{part}_leak"] = V.leak(s[m], tt, nn)
            r[f"{part}_leak_auc_t0"], r[f"{part}_leak_auc_t1"], r[f"{part}_leak_abs"] = V.leak_parts(s[m], tt, nn)
        m, tt, nn = ev["test"]
        dT = V.delta_t(s[m], tt, nn)
        r["test_deltaT"] = dT
        for kind, (a, bb, sg) in pairs.items():
            r[f"cf_{kind}"] = V.cf_effect(s[a], s[bb], sg, dT)
        if "cf_nuisance_rev" in r:  # Civil: average the two edit directions (CEBaB/BiasBios pairs are already bidirectional)
            r["cf_nuisance_fwd"] = r["cf_nuisance"]
            r["cf_nuisance"] = 0.5 * (r["cf_nuisance_fwd"] + r["cf_nuisance_rev"])
        # further counterfactual variants (absolute, per direction, validated subsets, placebos, other
        # normalizations) are computed post hoc from the saved scores in scripts/08_cf_variants.py
        rows.append(r)
        scores[(layer, rho, seed, name)] = s[ev["score_idx"]].astype(np.float32)
    return rows, scores


# labels of training rows are looked up from the global arrays bound below
def tr_t_lookup(tr, ev):
    return ev["T_all"][tr].astype(int)


def tr_n_lookup(tr, ev):
    return ev["N_all"][tr].astype(int)


def main(ds, model):
    texts = pd.read_parquet(ROOT / "data" / ds / "texts.parquet")
    pairs_df = pd.read_parquet(ROOT / "data" / ds / "pairs.parquet")
    A = np.load(ROOT / "data" / ds / f"acts_{model}.npz", allow_pickle=True)
    # rows appended to texts.parquet after extraction (e.g. new placebo edits) have no activations yet:
    # use the aligned prefix and drop pairs that reference the missing rows
    n_act = len(A["uid"])
    assert (A["uid"] == texts.uid.values[:n_act]).all()
    if n_act < len(texts):
        print(f"WARNING: {len(texts) - n_act} rows lack activations; their pairs are skipped", flush=True)
        texts = texts.iloc[:n_act]
        pairs_df = pairs_df[pairs_df.uid_a.isin(texts.uid) & pairs_df.uid_b.isin(texts.uid)]
    pos = {u: i for i, u in enumerate(texts.uid)}
    both = texts["T"].notna() & texts["N"].notna()
    T_all = texts["T"].fillna(-1).values
    N_all = texts["N"].fillna(-1).values
    tr_pool = np.where((texts.split == "train") & both)[0]
    ev = {"T_all": T_all, "N_all": N_all}
    for part in ("test", "dev"):
        m = np.where((texts.split == part) & both)[0]
        ev[part] = (m, T_all[m].astype(int), N_all[m].astype(int))
    pairs = {}
    for kind, g in pairs_df.groupby("kind"):
        pairs[kind] = (g.uid_a.map(pos).values, g.uid_b.map(pos).values, g.sign.values.astype(float))
    score_idx = np.where(texts.split.isin(["test", "cf", "cfsrc"]))[0]
    ev["score_idx"] = score_idx

    layers = A["layers"]
    tasks = []
    for li, layer in enumerate(layers):
        X = np.ascontiguousarray(A["X"][:, li, :])
        for rho in RHOS:
            for seed in SEEDS:
                tasks.append((X, li, int(layer), rho, seed))
    out = Parallel(n_jobs=18, verbose=5)(
        delayed(run_task)(X, li, layer, rho, seed, T_all[tr_pool].astype(int), N_all[tr_pool].astype(int),
                          tr_pool, ev, pairs)
        for X, li, layer, rho, seed in tasks)
    rows = [r for rs, _ in out for r in rs]
    scores = {}
    for _, sc in out:
        scores.update(sc)
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    df.insert(0, "dataset", ds)
    df.to_parquet(ROOT / "results" / f"pop_{ds}_{model}.parquet")
    keys = sorted(scores)
    np.savez_compressed(ROOT / "results" / f"scores_{ds}_{model}.npz",
                        S=np.stack([scores[k] for k in keys]),
                        keys=np.array([f"{k[0]}|{k[1]}|{k[2]}|{k[3]}" for k in keys]),
                        uid=texts.uid.values[score_idx])
    print(df.groupby(["rho", "family"])[["test_dec", "test_conflict", "test_leak", "cf_nuisance"]].mean().round(3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
