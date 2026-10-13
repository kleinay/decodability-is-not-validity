"""Placebo (neutral) edits, appended to the END of each texts.parquet (activation rows stay aligned).
  placebo        : swap target- and nuisance-neutral words for synonyms (all datasets). Changes the text and its
                   token sequence but neither T nor N -> the floor for |CF| any edit produces.
  placebo_group  : Civil only, 'Christian(s)' -> 'Protestant(s)' on the same 400 sources as the reverse
                   identity edits: an identity-term substitution that leaves N unchanged.
Pairs get sign +1 (no meaningful direction; the signed mean should be ~0). Idempotent."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEUTRAL = [("also", "additionally"), ("currently", "presently"), ("received", "obtained"), ("completed", "finished"),
           ("began", "started"), ("however", "nevertheless"), ("said", "stated"), ("think", "believe"),
           ("maybe", "perhaps"), ("often", "frequently"), ("usually", "typically"), ("several", "a number of"),
           ("large", "big"), ("near", "close to"), ("came", "arrived"), ("got", "obtained"), ("place", "spot"),
           ("went", "headed"), ("many", "numerous"), ("then", "afterwards"), ("because", "since"),
           ("about", "regarding"), ("just", "simply")]


def neutral_swap(t):
    for a, b in NEUTRAL:
        t = re.sub(r"\b" + a + r"\b", lambda m, b=b: b.capitalize() if m.group(0)[0].isupper() else b, t, flags=re.I)
    return t


def group_swap(t):
    for a, b in [("christians", "protestants"), ("christianity", "protestantism"), ("christian", "protestant")]:
        t = re.sub(r"\b" + a + r"\b", lambda m, b=b: b.capitalize() if m.group(0)[0].isupper() else b, t, flags=re.I)
    return t


rng = np.random.RandomState(7)
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    if (pr.kind == "placebo").any():
        print(ds, "already has placebos"); continue
    if ds == "cebab":
        src = tx[(tx.split == "test") & tx["T"].notna() & tx["N"].notna()]  # all labelled test reviews
    elif ds == "bios":
        src = tx[tx.uid.isin(pr.uid_a)]
    else:
        src = tx[tx.uid.isin(pr[pr.kind == "nuisance"].uid_a)]  # Muslim/LGBTQ-mentioning sources
    new = src.copy()
    new["text"] = new.text.map(neutral_swap)
    keep = new.text.values != src.text.values
    src, new = src[keep], new[keep].copy()
    new["uid"] = new.uid + "-plc"
    new["split"] = "cf"
    rows, pairs = [new], [pd.DataFrame({"uid_a": src.uid.values, "uid_b": new.uid.values, "kind": "placebo", "sign": 1})]
    msg = f"{ds}: placebo {len(new)}"
    if ds == "civil":
        gs = tx[tx.split == "cfsrc"]
        g = gs.copy()
        g["text"] = g.text.map(group_swap)
        k = g.text.values != gs.text.values
        gs, g = gs[k], g[k].copy()
        g["uid"] = g.uid + "-grp"
        g["split"] = "cf"
        rows.append(g)
        pairs.append(pd.DataFrame({"uid_a": gs.uid.values, "uid_b": g.uid.values, "kind": "placebo_group", "sign": 1}))
        msg += f", placebo_group {len(g)}"
    pd.concat([tx] + rows, ignore_index=True).to_parquet(ROOT / f"data/{ds}/texts.parquet")
    pd.concat([pr] + pairs, ignore_index=True).to_parquet(ROOT / f"data/{ds}/pairs.parquet")
    print(msg, "| e.g.:", src.text.iloc[0][:120], "->", new.text.iloc[0][:120])
