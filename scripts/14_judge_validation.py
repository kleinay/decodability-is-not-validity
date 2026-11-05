"""Validating the LLM judges without new human annotation.

(1) CEBaB human labels. CEBaB's edits were written by crowd workers and every text's food and service sentiment was
    labelled by 5 other workers. We judge (original, edit) pairs from the test and dev families with the same blind
    judges and compare each judge's 'service same?' (Q1) and 'food same?' (Q2) verdicts, and the accept rule, with the
    human majority labels. Human reference: one held-out worker per text against the majority of the other four.
(2) Planted errors. On top of edits both judges accepted (Amazon-Neg, Amazon-1P, BiasBios) we plant one known error
    and measure how often each judge, and the pair of judges, still accepts. LLM-planted errors are phrase
    replacements (gpt-oss-120b, JSON, exact match) and must keep the nuisance status by word list (Amazon); garbling
    and mixed gender cues are planted by rule. 'clean' re-judges unmodified accepted edits (fresh order: retest).

  python scripts/14_judge_validation.py build_cebab      -> data/cebab_gold/{texts,pairs}.parquet
  python scripts/14_judge_validation.py build_planted    -> data/<ds>_planted/{texts,pairs}.parquet  (Groq, < $0.40)
  (judging: 02c/02d for cebab_gold and bios_planted, 01i judgefull for amz*_planted; see gq_judge_validation.sh)
  python scripts/14_judge_validation.py analyze          -> results/rev_J_cebab_gold.csv, rev_J_planted.csv"""
import difflib
import zlib
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from dv.judging import accept  # noqa: E402

LAB = ["Positive", "Negative", "unknown"]


# ---------------------------------------------------------------- (1) CEBaB gold
def build_cebab():
    from datasets import load_dataset
    ds = load_dataset("CEBaB/CEBaB")
    d = pd.concat([ds[s].to_pandas() for s in ["test", "validation"]], ignore_index=True)
    d["uid"] = "cebabg-" + d["id"].astype(str)
    rows = []
    for _, fam in d.groupby("original_id"):
        o = fam[fam.is_original]
        if len(o) != 1:
            continue
        o = o.iloc[0]
        for _, e in fam[~fam.is_original].iterrows():
            if e.edit_type not in ("food", "service", "ambiance", "noise"):
                continue
            r = dict(uid_a=o.uid, uid_b=e.uid, edit_type=e.edit_type,
                     kind={"food": "nuisance", "service": "target"}.get(e.edit_type, "placebo"), sign=1)
            for asp in ["food", "service"]:
                r[f"{asp}_a"], r[f"{asp}_b"] = o[f"{asp}_aspect_majority"], e[f"{asp}_aspect_majority"]
                r[f"{asp}_wa"], r[f"{asp}_wb"] = o[f"{asp}_aspect_validation_workers"], e[f"{asp}_aspect_validation_workers"]
            rows.append(r)
    P = pd.DataFrame(rows)
    P = P[P[["food_a", "food_b", "service_a", "service_b"]].isin(LAB).all(1)].reset_index(drop=True)
    P["gold_food_same"], P["gold_serv_same"] = P.food_a == P.food_b, P.service_a == P.service_b
    # every gold-reject pair (edit did not change its aspect), plus a stratified sample of the rest
    rej = P[((P.kind == "nuisance") & P.gold_food_same) | ((P.kind == "target") & P.gold_serv_same)]
    rest = P.drop(rej.index)
    cap = {"nuisance": 220, "target": 220, "placebo": 160}
    samp = pd.concat([g.sample(n=min(cap[k], len(g)), random_state=0) for k, g in rest.groupby("kind")])
    P = pd.concat([rej, samp]).sort_index().reset_index(drop=True)
    out = ROOT / "data/cebab_gold"
    out.mkdir(exist_ok=True)
    tx = d[d.uid.isin(set(P.uid_a) | set(P.uid_b))][["uid", "description"]].rename(columns={"description": "text"})
    tx.assign(split="test", T=np.nan, N=np.nan).to_parquet(out / "texts.parquet")
    P.to_parquet(out / "pairs.parquet")
    print(len(P), P.groupby(["kind", "gold_serv_same", "gold_food_same"]).size().to_string())


def loo_human(P, rng):
    """One held-out worker per text vs the majority of the other four: same/different agreement with that majority."""
    res = {}
    for asp, gcol in [("food", "food"), ("service", "serv")]:
        agree = []
        for _, r in P.iterrows():
            wa, wb = eval(r[f"{asp}_wa"]), eval(r[f"{asp}_wb"])  # noqa: S307 (dataset's own dict strings)
            if len(wa) < 3 or len(wb) < 3:
                continue
            ka, kb = rng.choice(list(wa)), rng.choice(list(wb))
            ma = pd.Series([v for k, v in wa.items() if k != ka]).value_counts()
            mb = pd.Series([v for k, v in wb.items() if k != kb]).value_counts()
            if (len(ma) > 1 and ma.iloc[0] == ma.iloc[1]) or (len(mb) > 1 and mb.iloc[0] == mb.iloc[1]):
                continue  # no strict majority among the other four
            agree.append((wa[ka] == wb[kb]) == (ma.index[0] == mb.index[0]))
        res[gcol] = (np.mean(agree), len(agree))
    return res


# ---------------------------------------------------------------- (2) planted errors
_spec = importlib.util.spec_from_file_location("neg01h", ROOT / "scripts/01h_negation_edits.py")

AMZ_TASK = {
    "flip": "Change ONE short phrase so that the reviewer's overall opinion of the product becomes the OPPOSITE (a positive review becomes clearly negative, a negative one clearly positive), e.g. \"works great\" -> \"works poorly\", \"waste of money\" -> \"worth every penny\".",
    "strength": "Change ONE short phrase so that the opinion becomes clearly STRONGER or clearly WEAKER but keeps its direction (positive stays positive, negative stays negative), e.g. \"good\" -> \"absolutely outstanding\", \"terrible\" -> \"a bit disappointing\", \"I love it\" -> \"it's okay\".",
    "fact": "Change ONE factual detail (a number, a duration, a part of the product, a person, a place) without changing the opinion or its strength, e.g. \"two weeks\" -> \"two months\", \"the lid\" -> \"the handle\", \"for my son\" -> \"for my daughter\".",
}
BIOS_TASK = {
    "occupation": "Change ONE or two short phrases so that the person's occupation becomes {occ} instead (keep everything else).",
    "fact": "Change ONE factual detail (a university, a city, a number of years, a hospital, a specialty or degree) without changing the person's occupation or gender, e.g. \"Harvard\" -> \"Yale\", \"15 years\" -> \"5 years\".",
}
TAIL = """ Copy the original phrase EXACTLY as it appears; keep the result natural and fluent.

Return only JSON: {{"replacements": [{{"original": "...", "new": "..."}}]}}

Text:
{text}"""


def garble(text, rng):
    w = text.split()
    if len(w) < 6:
        return None
    i = rng.randint(1, len(w) - 2)
    w = w[:i] + [w[i + 1], w[i]] + w[i + 2:]          # swap two adjacent words
    j = rng.randint(0, len(w))
    w.insert(j, w[min(j, len(w) - 1)])                  # duplicate one word
    k = rng.randint(0, len(w))
    return " ".join(w[:k] + w[k + 1:])                  # drop one word


def mixed_gender(src, edt, rng):
    """Revert one (of several) gendered substitutions, leaving contradictory gender cues."""
    a, b = re.findall(r"\S+", src), re.findall(r"\S+", edt)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] == "replace" and (o[2] - o[1]) == (o[4] - o[3])]
    if len(ops) < 2:
        return None
    _, i1, i2, j1, j2 = ops[rng.randint(len(ops))]
    return " ".join(b[:j1] + a[i1:i2] + b[j2:])


def build_planted():
    H = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(H)
    H.BUDGET_USD = 0.40
    MODEL = "openai/gpt-oss-120b"
    rng = np.random.RandomState(7)
    RXFP = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves|i'm|i've|i'd|i'll|we're|we've)\b", re.I)
    RX = {"amzneg": re.compile(H.NEG.pattern, re.I), "amzfp": RXFP}
    for ds in ["amzneg", "amzfp", "bios"]:
        tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid")
        pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
        pr = pr[pr.kind == "nuisance"].reset_index(drop=True)
        ok = np.ones(len(pr), bool)
        for f in ["judge.parquet", "judge_groq.parquet"]:
            j = pd.read_parquet(ROOT / f"data/{ds}/{f}")
            acc = pd.Series(accept(j).values, index=pd.MultiIndex.from_frame(j[["uid_a", "uid_b"]]))
            ok &= acc.reindex(pd.MultiIndex.from_frame(pr[["uid_a", "uid_b"]])).fillna(False).values.astype(bool)
        pool = pr[ok].sample(frac=1, random_state=1).reset_index(drop=True)
        tasks = (["flip", "strength", "fact"] if ds != "bios" else ["occupation", "fact"]) + ["garble", "clean"] \
            + (["mixed_gender"] if ds == "bios" else [])
        per = 50
        jobs = []  # (ptype, source uid, edited uid, edited text)
        for ti, ptype in enumerate(tasks):
            for _, p in pool.iloc[ti * 70:(ti + 1) * 70].iterrows():   # disjoint sources per type; spare for failures
                jobs.append((ptype, p.uid_a, p.uid_b, tx.text[p.uid_b], tx.text[p.uid_a], tx["T"][p.uid_a]))

        def one(job):
            ptype, ua, ub, edt, src, t = job
            r = np.random.RandomState(zlib.crc32(f"{ptype}|{ub}".encode()))
            if ptype == "clean":
                return edt, "-"
            if ptype == "garble":
                return garble(edt, r), "-"
            if ptype == "mixed_gender":
                return mixed_gender(src, edt, r), "-"
            if ds == "bios":
                task = BIOS_TASK[ptype].format(occ="a nurse" if t == 1 else "a surgeon")
            else:
                task = AMZ_TASK[ptype]
            out = H._post(MODEL, {**H._body(MODEL, task + TAIL.format(text=edt), json_mode=True), "reasoning_effort": "medium"})
            return H.apply_replacements(edt, out) if out else (None, "empty")
        with ThreadPoolExecutor(6) as ex:
            outs = list(ex.map(one, jobs))
        rows, texts = [], []
        for (ptype, ua, ub, edt, src, t), (new, why) in zip(jobs, outs):
            if not new or (ptype != "clean" and new == edt):
                continue
            if ds in RX and bool(RX[ds].search(new)) != bool(RX[ds].search(edt)):
                continue  # planted error must not undo the nuisance flip (that is checked by word list, not judged)
            if sum(r["ptype"] == ptype for r in rows) >= per:
                continue
            nu = f"{ub}-pl-{ptype}"
            rows.append(dict(uid_a=ua, uid_b=nu, kind="nuisance", sign=1, ptype=ptype, uid_orig_edit=ub))
            texts.append(dict(uid=nu, text=new))
        P = pd.DataFrame(rows)
        T = pd.concat([tx.loc[P.uid_a.unique()].reset_index()[["uid", "text"]], pd.DataFrame(texts)], ignore_index=True)
        T = T.drop_duplicates("uid").assign(split="test", T=np.nan, N=np.nan)
        out = ROOT / f"data/{ds}_planted"
        out.mkdir(exist_ok=True)
        T.to_parquet(out / "texts.parquet")
        P.to_parquet(out / "pairs.parquet")
        print(ds, P.ptype.value_counts().to_dict(), f"spend ${H.spent:.3f}", flush=True)


# ---------------------------------------------------------------- analysis
def verdict_table(ds):
    P = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    out = P.copy()
    for tag, f in [("j1", "judge.parquet"), ("j2", "judge_groq.parquet")]:
        j = pd.read_parquet(ROOT / f"data/{ds}/{f}")
        j = j.assign(acc=accept(j).values).set_index(["uid_a", "uid_b"])
        k = pd.MultiIndex.from_frame(P[["uid_a", "uid_b"]])
        out[f"{tag}_acc"] = j.acc.reindex(k).fillna(False).values.astype(bool)
        for c in ["target_same", "nuis_same"]:
            out[f"{tag}_{c}"] = j[c].reindex(k).astype(str).values
    out["both_acc"] = out.j1_acc & out.j2_acc
    return out


def analyze():
    rng = np.random.RandomState(0)
    # (1) CEBaB: per question and per accept decision, judges vs human majority labels
    V = verdict_table("cebab_gold")
    V["gold_acc"] = np.select([V.kind.eq("nuisance"), V.kind.eq("target"), V.kind.eq("placebo")],
                              [V.gold_serv_same & ~V.gold_food_same, ~V.gold_serv_same & V.gold_food_same,
                               V.gold_serv_same & V.gold_food_same], False)
    hum = loo_human(V, rng)
    rows = []
    for tag in ["j1", "j2"]:
        for q, gcol, h in [("target_same", "gold_serv_same", "serv"), ("nuis_same", "gold_food_same", "food")]:
            said_same = V[f"{tag}_{q}"].str.startswith("yes")
            said_diff = V[f"{tag}_{q}"].str.startswith("no")
            g = V[gcol]
            rows.append(dict(judge=tag, question=q, n=len(V), acc=((said_same & g) | (said_diff & ~g)).mean(),
                             hit_diff=said_diff[~g].mean(), hit_same=said_same[g].mean(), n_gold_diff=int((~g).sum()),
                             human_single_vs_majority=hum[h][0], human_n=hum[h][1]))
    for tag in ["j1", "j2", "both"]:
        a, g = V[f"{tag}_acc"], V.gold_acc
        # 'edits' = nuisance + target edits. For placebo (ambiance/noise) edits the human labels cover food and service
        # only, while the judges are also asked about other meaning changes, which those edits do make.
        for kind in ["nuisance", "target", "placebo", "edits"]:
            m = V.kind.isin(["nuisance", "target"]) if kind == "edits" else (V.kind == kind)
            rows.append(dict(judge=tag, question=f"accept_{kind}", n=int(m.sum()),
                             acc=(a[m] == g[m]).mean(), recall_gold_ok=a[m & g].mean() if (m & g).any() else np.nan,
                             reject_gold_bad=(~a[m & ~g]).mean() if (m & ~g).any() else np.nan,
                             precision=g[m & a].mean() if (m & a).any() else np.nan, n_gold_bad=int((m & ~g).sum())))
    R1 = pd.DataFrame(rows)
    R1.to_csv(RES / "rev_J_cebab_gold.csv", index=False)
    pd.set_option("display.width", 250)
    print(R1.round(3).to_string(index=False))
    # (2) planted errors: share accepted per type, per judge
    rows = []
    for ds in ["amzneg", "amzfp", "bios"]:
        V = verdict_table(f"{ds}_planted")
        for ptype, g in V.groupby("ptype"):
            rows.append(dict(dataset=ds, ptype=ptype, n=len(g), j1_accept=g.j1_acc.mean(), j2_accept=g.j2_acc.mean(),
                             both_accept=g.both_acc.mean()))
    R2 = pd.DataFrame(rows)
    R2.to_csv(RES / "rev_J_planted.csv", index=False)
    print(R2.round(3).to_string(index=False))


if __name__ == "__main__":
    {"build_cebab": build_cebab, "build_planted": build_planted, "analyze": analyze}[sys.argv[1]]()
