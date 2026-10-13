"""Reverse-direction identity edits for Civil (Christian -> Muslim), so that the Civil counterfactual
effect can be estimated in both directions like CEBaB and BiasBios (where non-directional artifacts of
editing cancel in the signed mean). Appends rows to the END of data/civil/texts.parquet (existing
activation rows stay aligned) and adds pairs of kind 'nuisance_rev'. Idempotent."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
texts = pd.read_parquet(ROOT / "data/civil/texts.parquet")
pairs = pd.read_parquet(ROOT / "data/civil/pairs.parquet")
if (texts.split == "cfsrc").any():
    sys.exit("already appended")

REV = [(r"christians", "muslims"), (r"christianity", "islam"), (r"christian", "muslim")]


def rev_swap(t):
    for pat, new in REV:
        t = re.sub(r"\b" + pat + r"\b", lambda m, new=new: (new.capitalize() if m.group(0)[0].isupper() else new), t, flags=re.I)
    return t


d = load_dataset("pietrolesci/civilcomments-wilds")["test"].to_pandas()
d = d[(d.christian == 1) & (d.muslim == 0) & (d.LGBTQ == 0) & (d.comment_text.str.split().str.len() >= 5)]
d = d[d.comment_text.map(rev_swap) != d.comment_text]
used = set(texts.uid)
d = d[~("civil-test-" + d["id"].astype(str)).isin(used)]
rng = np.random.RandomState(1)
d = pd.concat([g.sample(n=min(200, len(g)), random_state=rng) for _, g in d.groupby("toxicity")])
src = pd.DataFrame({"uid": "civil-rsrc-" + d["id"].astype(str), "text": d.comment_text.values, "split": "cfsrc",
                    "T": d.toxicity.astype(float).values, "N": 0.0})
cf = src.copy()
cf["text"] = cf.text.map(rev_swap)
cf["uid"] = cf.uid + "-cf"
cf["split"] = "cf"
cf["N"] = 1.0
assert (cf.text.values != src.text.values).all()
new_pairs = pd.DataFrame({"uid_a": src.uid.values, "uid_b": cf.uid.values, "kind": "nuisance_rev", "sign": 1})
pd.concat([texts, src, cf], ignore_index=True).to_parquet(ROOT / "data/civil/texts.parquet")
pd.concat([pairs, new_pairs], ignore_index=True).to_parquet(ROOT / "data/civil/pairs.parquet")
print(f"appended {len(src)} sources + {len(cf)} edits; toxic share {src['T'].mean():.2f}")
print(src.text.iloc[0][:200], "\n->", cf.text.iloc[0][:200])
