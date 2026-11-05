"""Build the two Amazon datasets in the unified schema (only these; does not touch cebab/bios/civil, whose texts carry
appended edits). Short reviews (8-40 words) of amazon_polarity; T = negative sentiment.
  amzneg : N = contains a negation word       (natural P(N|T=1)=.64, P(N|T=0)=.30: correlated)
  amzfp  : N = reviewer refers to themselves  (natural .66 vs .67: uncorrelated)
train/dev come from disjoint slices of the amazon_polarity train split (it has no dev split), test from its test
split; exact-duplicate texts across splits are dropped. Per (T, N) cell at most 1100/250/400 train/dev/test texts.
Writes data/<ds>/{texts,pairs}.parquet (pairs empty: edits are appended by 01i full) and data/amz/natural_rates.json."""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
NEG = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|nowhere|cannot|without|hardly|barely)\b|n't\b", re.I)
FP = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves|i'm|i've|i'd|i'll|we're|we've)\b", re.I)
CAPS = {"train": 1100, "dev": 250, "test": 400}


def short(df, name):
    df = df.copy()
    df["text"] = df.content.str.strip()
    df = df[df.text.str.split().str.len().between(8, 40)]
    df["uid"] = f"amz-{name}-" + df.index.astype(str)
    df["T"] = (1 - df.label).astype(float)
    df["NEG"] = df.text.map(lambda s: float(bool(NEG.search(s))))
    df["FP"] = df.text.map(lambda s: float(bool(FP.search(s))))
    return df[["uid", "text", "T", "NEG", "FP"]]


tr_stream = list(itertools.islice(load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True), 150000))
train = short(pd.DataFrame(tr_stream[:120000]), "train")
dev = short(pd.DataFrame(tr_stream[120000:]).reset_index(drop=True), "dev")
test = pd.read_parquet(ROOT / "data/amz/pool_test_short.parquet")[["uid", "text", "T", "NEG", "FP"]].astype({"T": float})
test["NEG"], test["FP"] = test.NEG.astype(float), test.FP.astype(float)
seen = set()
parts = {}
for name, d in [("train", train), ("dev", dev), ("test", test)]:
    d = d[~d.text.isin(seen)].drop_duplicates("text")
    seen |= set(d.text)
    parts[name] = d
nat = {f"P({v}|{lab})": float(parts["train"][parts["train"]["T"] == t][v].mean())
       for v in ["NEG", "FP"] for t, lab in [(1, "negative"), (0, "positive")]}
(ROOT / "data/amz/natural_rates.json").write_text(json.dumps(nat, indent=1))
print("natural rates (train):", nat)

rng = np.random.RandomState(0)
for ds, col in [("amzneg", "NEG"), ("amzfp", "FP")]:
    frames = []
    for name, d in parts.items():
        d = d.assign(N=d[col], split=name)
        for (t, n), g in d.groupby(["T", "N"]):
            frames.append(g.sample(n=min(CAPS[name], len(g)), random_state=rng))
    texts = pd.concat(frames, ignore_index=True)[["uid", "text", "split", "T", "N"]]
    assert texts.uid.is_unique
    out = ROOT / "data" / ds
    out.mkdir(exist_ok=True)
    if (out / "texts.parquet").exists():
        sys.exit(f"{out}/texts.parquet exists; refusing to overwrite (edits may have been appended)")
    texts.to_parquet(out / "texts.parquet")
    pd.DataFrame(columns=["uid_a", "uid_b", "kind", "sign"]).astype({"sign": int}).to_parquet(out / "pairs.parquet")
    print(ds, texts.groupby(["split", "T", "N"]).size().unstack([1, 2]).to_string())
