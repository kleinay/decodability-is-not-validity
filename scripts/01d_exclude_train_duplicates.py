"""Exclude dev/test texts whose exact string also occurs in the training split (found by 07_audit.py:
2 BiasBios test texts, 1 dev). Rows are relabelled split='dup_excluded' in place, so activation files
stay row-aligned; counterfactual pairs built from them are dropped. Idempotent."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    train = set(tx[tx.split == "train"].text)
    dup = tx.split.isin(["dev", "test", "cfsrc"]) & tx.text.isin(train) & tx["T"].notna() & tx["N"].notna()
    if dup.any():
        bad = set(tx.uid[dup])
        tx.loc[dup, "split"] = "dup_excluded"
        n0 = len(pr)
        pr = pr[~pr.uid_a.isin(bad) & ~pr.uid_b.isin(bad)]
        tx.to_parquet(ROOT / f"data/{ds}/texts.parquet")
        pr.to_parquet(ROOT / f"data/{ds}/pairs.parquet")
        print(f"{ds}: excluded {dup.sum()} texts, dropped {n0 - len(pr)} pairs")
