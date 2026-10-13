"""Unified (target T, nuisance N) view of three public datasets.

Every dataset is reduced to the same schema so one pipeline handles all three:

    texts.parquet : uid, text, split in {train, dev, test, cf}, T (0/1/NaN), N (0/1/NaN)
    pairs.parquet : uid_a, uid_b, kind in {nuisance, target}, sign (+1 if b moves the
                    flipped variable toward 1, -1 toward 0)

Datasets (each a different *type* of variable pair):
  cebab : T = SERVICE sentiment, N = FOOD sentiment          (two evaluative aspects of one text)
  bios  : T = surgeon vs. nurse,  N = male vs. female subject (occupation vs. demographic attribute)
  civil : T = toxic vs. not,      N = mentions Muslim/LGBTQ identity (pragmatic property vs. topic)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from datasets import load_dataset

SEED = 0


def _cap_cells(df: pd.DataFrame, cap: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Keep at most `cap` rows per (T, N) cell -- gives every cell enough rows for
    reweighting to any T-N correlation without extracting the full pool."""
    out = []
    for _, g in df.groupby(["T", "N"]):
        out.append(g.sample(n=min(cap, len(g)), random_state=rng))
    return pd.concat(out).sort_index()


# ---------------------------------------------------------------- CEBaB
def build_cebab():
    ds = load_dataset("CEBaB/CEBaB")
    lab = {"Positive": 1.0, "Negative": 0.0}
    frames = []
    for split, name in [("train_inclusive", "train"), ("validation", "dev"), ("test", "test")]:
        d = ds[split].to_pandas()
        d = pd.DataFrame({
            "uid": "cebab-" + d["id"].astype(str),
            "text": d["description"],
            "split": name,
            "T": d["service_aspect_majority"].map(lab),
            "N": d["food_aspect_majority"].map(lab),
            "family": d["original_id"].astype(str),
            "edit_type": d["edit_type"].astype(str),
            "food_raw": d["food_aspect_majority"],
            "service_raw": d["service_aspect_majority"],
        })
        frames.append(d)
    texts = pd.concat(frames, ignore_index=True)
    assert not set(texts.query("split=='train'").family) & set(texts.query("split!='train'").family)

    # Counterfactual pairs inside the TEST split: within a family, two texts that differ
    # in the edited aspect's Pos/Neg label while the other aspect's raw label is unchanged.
    test = texts[texts.split == "test"]
    pairs = []
    for _, fam in test.groupby("family"):
        orig = fam[fam.edit_type.isin(["None", "nan"])]
        if len(orig) != 1:
            continue
        o = orig.iloc[0]
        for _, e in fam.iterrows():
            if e.uid == o.uid:
                continue
            if e.edit_type == "food" and e.service_raw == o.service_raw:
                if not np.isnan(e.N) and not np.isnan(o.N) and e.N != o.N:
                    pairs.append((o.uid, e.uid, "nuisance", 1 if e.N > o.N else -1))
            if e.edit_type == "service" and e.food_raw == o.food_raw:
                if not np.isnan(e["T"]) and not np.isnan(o["T"]) and e["T"] != o["T"]:
                    pairs.append((o.uid, e.uid, "target", 1 if e["T"] > o["T"] else -1))
    pairs = pd.DataFrame(pairs, columns=["uid_a", "uid_b", "kind", "sign"])
    texts = texts[["uid", "text", "split", "T", "N"]]
    return texts, pairs


# ---------------------------------------------------------------- Bias in Bios
_GENDER_SWAP = {
    "he": "she", "she": "he", "him": "her", "his": "her", "hers": "his",
    "himself": "herself", "herself": "himself", "mr": "ms", "mrs": "mr", "ms": "mr",
    "man": "woman", "woman": "man", "men": "women", "women": "men",
    "husband": "wife", "wife": "husband", "father": "mother", "mother": "father",
    "son": "daughter", "daughter": "son", "sons": "daughters", "daughters": "sons",
    "boyfriend": "girlfriend", "girlfriend": "boyfriend", "brother": "sister", "sister": "brother",
    "dad": "mom", "mom": "dad", "male": "female", "female": "male",
}
_HER_OBJ_NEXT = re.compile(r"^\s*([.,;:!?)\"']|$|(and|or|to|in|on|at|for|with|as|from|by|about|that|the|a|an)\b)", re.I)


def gender_swap(text: str) -> str:
    """Rule-based pronoun/kinship swap. 'her' is resolved to 'his' (possessive) unless the
    next token looks like the end of a noun phrase, in which case 'him' (object)."""
    def repl(m):
        w = m.group(0)
        lw = w.lower()
        if lw == "her":
            rest = text[m.end():]
            new = "him" if _HER_OBJ_NEXT.match(rest) else "his"
        else:
            new = _GENDER_SWAP[lw]
        if w.isupper() and len(w) > 1:
            return new.upper()
        if w[0].isupper():
            return new[0].upper() + new[1:]
        return new
    pat = r"\b(" + "|".join(sorted(set(_GENDER_SWAP) | {"her"}, key=len, reverse=True)) + r")\b"
    return re.sub(pat, repl, text, flags=re.I)


def build_bios(cap_train=1100, cap_dev=250, cap_test=400):
    ds = load_dataset("LabHC/bias_in_bios")
    SURGEON, NURSE = 25, 13
    rng = np.random.RandomState(SEED)
    frames = []
    for split, name, cap in [("train", "train", cap_train), ("dev", "dev", cap_dev), ("test", "test", cap_test)]:
        d = ds[split].to_pandas()
        d = d[d.profession.isin([SURGEON, NURSE])]
        d = pd.DataFrame({
            "uid": f"bios-{name}-" + d.index.astype(str),
            "text": d["hard_text"],
            "split": name,
            "T": (d.profession == SURGEON).astype(float),
            "N": (d.gender == 0).astype(float),  # 1 = male subject
        })
        frames.append(_cap_cells(d, cap, rng))
    texts = pd.concat(frames, ignore_index=True)
    test = texts[texts.split == "test"]
    cf = test.copy()
    cf["text"] = cf.text.map(gender_swap)
    changed = cf.text.values != test.text.values
    cf = cf[changed]
    cf["uid"] = cf.uid + "-cf"
    cf["split"] = "cf"
    cf["N"] = 1.0 - cf["N"]
    pairs = pd.DataFrame({
        "uid_a": test.uid.values[changed], "uid_b": cf.uid.values, "kind": "nuisance",
        "sign": np.where(cf.N.values > test.N.values[changed], 1, -1),
    })
    return pd.concat([texts, cf], ignore_index=True), pairs


# ---------------------------------------------------------------- CivilComments (WILDS)
_IDENTITY_SWAP = [
    (r"muslims", "christians"), (r"muslim", "christian"), (r"moslems?", "christian"),
    (r"islamic", "christian"), (r"islamists?", "fundamentalist"), (r"islam", "christianity"),
    (r"homosexuality", "heterosexuality"), (r"homosexuals", "heterosexuals"), (r"homosexual", "heterosexual"),
    (r"lesbians", "straight women"), (r"lesbian", "straight"), (r"gays", "straight people"), (r"gay", "straight"),
    (r"transgenders?", "cisgender"), (r"bisexuals?", "heterosexual"), (r"queer", "straight"),
    (r"lgbtq?\+?", "straight"),
]


def identity_swap(text: str) -> str:
    def make_repl(new):
        def repl(m):
            w = m.group(0)
            if w.isupper() and len(w) > 1:
                return new.upper()
            if w[0].isupper():
                return new[0].upper() + new[1:]
            return new
        return repl
    for pat, new in _IDENTITY_SWAP:
        text = re.sub(r"\b" + pat + r"\b", make_repl(new), text, flags=re.I)
    return text


def build_civil(cap_train=1100, cap_dev=250, cap_test=400):
    ds = load_dataset("pietrolesci/civilcomments-wilds")
    rng = np.random.RandomState(SEED)
    frames = []
    for split, name, cap in [("train", "train", cap_train), ("validation", "dev", cap_dev), ("test", "test", cap_test)]:
        d = ds[split].to_pandas()
        d = d[d.comment_text.str.split().str.len() >= 5]
        d = pd.DataFrame({
            "uid": f"civil-{name}-" + d["id"].astype(str),
            "text": d["comment_text"],
            "split": name,
            "T": d["toxicity"].astype(float),
            "N": ((d["muslim"] == 1) | (d["LGBTQ"] == 1)).astype(float),
        })
        frames.append(_cap_cells(d, cap, rng))
    texts = pd.concat(frames, ignore_index=True)
    test = texts[(texts.split == "test") & (texts.N == 1)]
    cf = test.copy()
    cf["text"] = cf.text.map(identity_swap)
    changed = cf.text.values != test.text.values
    cf = cf[changed]
    cf["uid"] = cf.uid + "-cf"
    cf["split"] = "cf"
    cf["N"] = 0.0
    pairs = pd.DataFrame({
        "uid_a": test.uid.values[changed], "uid_b": cf.uid.values, "kind": "nuisance", "sign": -1,
    })
    return pd.concat([texts, cf], ignore_index=True), pairs


BUILDERS = {"cebab": build_cebab, "bios": build_bios, "civil": build_civil}
