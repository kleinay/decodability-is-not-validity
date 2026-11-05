"""Candidate replacement for Civil: short Amazon reviews (amazon_polarity test, 8-40 words), target T = negative
sentiment, and TWO lexically verifiable nuisances edited on the same pool:
  neg : contains a negation word        (naturally correlated with T: P = .64 neg vs .30 pos)
  fp  : the reviewer refers to themselves (first person; naturally uncorrelated: .66 vs .67)
An LLM proposes phrase replacements (JSON), applied by exact match (as in 01h). The nuisance flip is verified by
regex; the two blind judges check sentiment kept, opinion strength kept (Q6, added to the 02c prompt), no major
other meaning change and no fluency drop.

  python scripts/01i_amazon_edits.py pilot <neg|fp> <model> [n_per_cell] [seed] -> data/amz/pilot_<nuis>_<model>_seed<k>.csv
  python scripts/01i_amazon_edits.py judge <csv> <neg|fp> <judge_model>          -> <csv>_judged_<judge>.csv"""
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("neg01h", ROOT / "scripts/01h_negation_edits.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

RX = {"neg": H.NEG,
      "fp": re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves|i'm|i've|i'd|i'll|we're|we've)\b", re.I)}
NEG_RX = re.compile(H.NEG.pattern, re.I)
RX["neg"] = NEG_RX

COMMON = """Each replacement must read as natural English that the same reviewer could have written, keep the reviewer's opinion of the product exactly as positive or negative and exactly as strong as before (not weaker, stronger, more hedged or mixed), keep every fact, and change nothing else. Keep the reviewer's spelling, capitalization and punctuation style. Copy each original phrase EXACTLY as it appears in the review. The new phrase must fit grammatically where it stands (include a preceding article or verb in the original phrase if it must change too). If no replacement satisfies all of this, return {{"skip": true}}.

Return only JSON: {{"replacements": [{{"original": "...", "new": "..."}}]}}

Review:
{text}"""

PROMPTS = {
    ("neg", 0): """We need a version of the product review below that contains a negation (not, n't, no, never, nothing, without, ...) but means exactly the same thing.
Choose ONE short phrase and give a replacement for it that contains a negation and is an exact paraphrase: e.g. "flimsy" -> "not sturdy", "stopped working" -> "no longer works", "always works" -> "never fails", "failed to" -> "did not", "lacks" -> "has no", "avoid it" -> "don't buy it", "zero problems" -> "no problems". Do NOT add intensifiers or emphasis ("nothing short of", "nothing but", "nothing less than", "not at all"). Avoid litotes and double negatives ("not unhappy", "never unreliable", "not without") and awkward antonyms ("not small", "not old", "not light"). Never just insert "not" in front of an unchanged phrase: that reverses its meaning.
""" + COMMON,
    ("neg", 1): """We need a version of the product review below that contains NO negation at all (no not, n't, no, never, nothing, nobody, none, neither, nor, nowhere, cannot, without, hardly, barely) but means exactly the same thing.
Give a replacement for EVERY phrase that contains one of these words, each an exact un-negated paraphrase: e.g. "doesn't work" -> "fails", "not worth the money" -> "overpriced", "never had a problem" -> "had zero problems", "don't buy" -> "avoid", "no complaints" -> "zero complaints". Never just delete the negation word: that reverses the meaning.
""" + COMMON,
    ("fp", 0): """We need a version of the product review below in which the reviewer refers to themselves (first person: I, me, my, we, our) but which means exactly the same thing.
Choose ONE short phrase and rephrase it in the first person without adding any new fact, hedge or emphasis. Best: restore a dropped subject or use a possessive where the reviewer's own action or item is already implied, e.g. "Would recommend" -> "I would recommend", "Purchased this for my trip" -> "I purchased this for my trip", "the battery died after a week" -> "my battery died after a week", "Returned it." -> "I returned it.". Do NOT add hedges, opinion markers or relativizers that make the claim personal or weaker ("I think", "I feel", "in my opinion", "I found", "for me", "to me"); do NOT turn generic "you" into "I"; never write "the reviewer".
""" + COMMON,
    ("fp", 1): """We need a version of the product review below in which the reviewer NEVER refers to themselves (no I, me, my, mine, myself, we, us, our, ours) but which means exactly the same thing.
Give a replacement for EVERY phrase that contains one of these words, each an impersonal paraphrase that keeps every fact and the same strength of opinion: e.g. "I love this blender" -> "this blender is lovable" is BAD (unnatural) but "I love this blender" -> "this blender is wonderful" is good, "my kids enjoy it" -> "kids enjoy it", "I would not buy it again" -> "not worth buying again", "it broke after I used it twice" -> "it broke after two uses", "we were disappointed" -> "disappointing". Never write "the reviewer" or "the buyer" as a stand-in for the writer; drop the subject instead where natural ("I returned it." -> "Returned it.").
""" + COMMON,
}

JUDGE_T = "the reviewer's overall opinion of the product (positive or negative)"
JUDGE_N = {"neg": ("whether the text contains an explicit negation word (not, n't, no, never, nothing, none, nobody, nowhere, "
                   "neither, nor, without, cannot, hardly, barely); words like 'fails', 'lacks' or 'un-' prefixes do NOT count"),
           "fp": "whether the writer refers to themselves (I, me, my, we, us, our)"}
Q6 = "Q6. Is the STRENGTH of the reviewer's opinion (how strongly positive or negative, how hedged) the same in Text 1 and Text 2? (yes / no)\n"


def edit(model, src, nuis):
    def one(r):
        b = H._body(model, PROMPTS[(nuis, int(r[nuis.upper()]))].format(text=r.text), json_mode=True)
        if "gpt-oss" in model:
            b["reasoning_effort"] = "medium"
        out = H._post(model, b)
        return H.apply_replacements(r.text, out) if out else (None, "empty")
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(one, [r for _, r in src.iterrows()]))
    rows = []
    for (_, r), (o, why) in zip(src.iterrows(), outs):
        n_src = int(r[nuis.upper()])
        c = dict(ok=False, why=why, frac_changed=np.nan)
        if o:
            c = H.checks(r.text, o, n_src)  # uses NEG; recompute the flip with the right regex
            flipped = bool(RX[nuis].search(o)) != bool(n_src)
            why = [w for w in c["why"].split("; ") if w not in ("-", "negation not flipped")]
            if not flipped:
                why.append("nuisance not flipped")
            c.update(ok=not why, why="; ".join(why) or "-")
        rows.append(dict(uid=r.uid, T=r["T"], N=n_src, source=r.text, edited=o, **c))
    return pd.DataFrame(rows)


def judge(df, nuis, model):
    from dv.judging import accept
    src = (ROOT / "scripts/02c_judge_edits.py").read_text()
    PROMPT = re.search(r'PROMPT = """(.*?)"""', src, re.S).group(1)
    PROMPT = PROMPT.replace("\n\nReply with only a JSON object", "\n" + Q6 + "\nReply with only a JSON object")
    PROMPT = PROMPT.replace('"q5": ...}}', '"q5": ..., "q6": ...}}')
    assert "q6" in PROMPT and "Q6." in PROMPT
    swaps = np.random.RandomState(0).rand(len(df)) < 0.5

    def one(i):
        r = df.iloc[i]
        a, b = (r.edited, r.source) if swaps[i] else (r.source, r.edited)
        out = H._post(model, H._body(model, PROMPT.format(a=a, b=b, t=JUDGE_T, n=JUDGE_N[nuis], note=""), json_mode=True))
        try:
            return {k: str(v).lower() for k, v in json.loads(out or "{}").items()}
        except Exception:
            return {}
    with ThreadPoolExecutor(6) as ex:
        js = list(ex.map(one, range(len(df))))
    J = df.copy()
    for col, q in [("target_same", "q1"), ("nuis_same", "q2"), ("other_change", "q5"), ("strength_same", "q6")]:
        J[col] = [j.get(q) for j in js]
    J["src_fluent"] = [j.get("q4" if s else "q3") for j, s in zip(js, swaps)]
    J["edit_fluent"] = [j.get("q3" if s else "q4") for j, s in zip(js, swaps)]
    J["kind"] = "nuisance"
    J["accepted_std"] = accept(J).values  # the main pipeline's rule (uses the judge's nuisance verdict)
    y = lambda c: J[c].astype(str).str.startswith("yes")  # noqa: E731
    n = lambda c: J[c].astype(str).str.startswith("no")  # noqa: E731
    # nuisance flip is verified by regex (J.ok); the judges decide sentiment, strength, meaning and fluency
    J["accepted"] = (J.ok & y("target_same") & y("strength_same") & ~J.other_change.astype(str).str.startswith("major")
                     & (y("edit_fluent") | n("src_fluent")))
    return J


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pilot":
        nuis, model = sys.argv[2], sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 25
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 13   # 13: prompt development; another seed held out
        pool = pd.read_parquet(ROOT / "data/amz/pool_test_short.parquet")
        src = pool.groupby(["T", nuis.upper()]).sample(n=n, random_state=seed)
        df = edit(model, src, nuis)
        out = ROOT / f"data/amz/pilot_{nuis}_{model.split('/')[-1]}_seed{seed}.csv"
        df.to_csv(out, index=False)
        print(f"{nuis} {model}: {len(df)} edits, auto checks passed {df.ok.mean():.0%} "
              f"(add {df[df.N == 0].ok.mean():.0%}, remove {df[df.N == 1].ok.mean():.0%}); spend ${H.spent:.3f}")
        print(df.why.value_counts().head(6).to_string())
    elif mode == "full":   # python 01i full <amzneg|amzfp> <model> <budget_usd>
        ds, model = sys.argv[2], sys.argv[3]
        H.BUDGET_USD = float(sys.argv[4])
        nuis = {"amzneg": "neg", "amzfp": "fp"}[ds.split("_")[0]]  # also *_planted (14_judge_validation.py)
        tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet")
        pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
        if len(pr):
            sys.exit(f"{ds}: pairs already exist")
        test = tx[tx.split == "test"].copy()
        test[nuis.upper()] = test["N"].astype(int)
        ep = ROOT / f"data/{ds}/llm_edits_{model.split('/')[-1]}.parquet"
        if ep.exists():   # reuse paid-for edits
            ed = pd.read_parquet(ep)
        else:
            ed = edit(model, test, nuis)
            ed.to_parquet(ep)
        good = ed[ed.ok & ed.edited.notna() & (ed.edited != ed.source)]
        new = pd.DataFrame({"uid": good.uid + "-llm", "text": good.edited.values, "split": "cf",
                            "T": good["T"].values.astype(float), "N": 1.0 - good["N"].values})
        pairs = pd.DataFrame({"uid_a": good.uid.values, "uid_b": new.uid.values, "kind": "nuisance",
                              "sign": np.where(new.N.values == 1, 1, -1)})
        # placebo: neutral synonym swaps (same word list as 01c) on test texts, nuisance status unchanged by regex
        src01c = (ROOT / "scripts/01c_placebo_edits.py").read_text()
        ns = {"__file__": str(ROOT / "scripts/01c_placebo_edits.py")}
        exec(src01c[src01c.index("import re"):src01c.index("def group_swap")], ns)  # noqa: S102
        pl = test.copy()
        pl["text"] = pl.text.map(ns["neutral_swap"])
        keep = (pl.text.values != test.text.values) & np.array(
            [bool(RX[nuis].search(a)) == bool(RX[nuis].search(b)) for a, b in zip(test.text, pl.text)])
        psrc, pl = test[keep], pl[keep].copy()
        pl["uid"] = pl.uid + "-plc"
        pl["split"] = "cf"
        ppairs = pd.DataFrame({"uid_a": psrc.uid.values, "uid_b": pl.uid.values, "kind": "placebo", "sign": 1})
        pd.concat([tx, new, pl[["uid", "text", "split", "T", "N"]]], ignore_index=True).to_parquet(ROOT / f"data/{ds}/texts.parquet")
        pd.concat([pairs, ppairs], ignore_index=True).to_parquet(ROOT / f"data/{ds}/pairs.parquet")
        print(f"{ds}: {len(ed)} edits generated, {len(good)} pass automatic checks "
              f"(add {ed[ed.N == 0].ok.mean():.0%}, remove {ed[ed.N == 1].ok.mean():.0%}); placebo {len(pl)}; spend ${H.spent:.3f}")
        print(ed.why.value_counts().head(6).to_string())
    elif mode == "judgefull":   # python 01i judgefull <amzneg|amzfp> <judge_model> <out: judge|judge_groq> <budget>
        ds, model, outname = sys.argv[2], sys.argv[3], sys.argv[4]
        H.BUDGET_USD = float(sys.argv[5])
        nuis = {"amzneg": "neg", "amzfp": "fp"}[ds.split("_")[0]]  # also *_planted (14_judge_validation.py)
        tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid")
        pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
        df = pd.DataFrame({"uid": pr.uid_a.values, "uid_a": pr.uid_a.values, "uid_b": pr.uid_b.values, "kind_": pr.kind.values,
                           "sign": pr.sign.values, "source": tx.text.reindex(pr.uid_a).values,
                           "edited": tx.text.reindex(pr.uid_b).values, "T": tx["T"].reindex(pr.uid_a).values,
                           "N": tx["N"].reindex(pr.uid_a).values})
        df["ok"] = True
        J = judge(df, nuis, model)
        J["kind"] = J.kind_
        flip = np.array([bool(RX[nuis].search(a)) != bool(RX[nuis].search(b)) for a, b in zip(J.source, J.edited)])
        J["nuis_flip_regex"] = flip
        J["judge_model"] = model
        cols = ["uid_a", "uid_b", "kind", "sign", "target_same", "nuis_same", "other_change", "src_fluent", "edit_fluent",
                "strength_same", "nuis_flip_regex", "judge_model"]
        J[cols].to_parquet(ROOT / f"data/{ds}/{outname}.parquet")
        from dv.judging import accept
        J["acc"] = accept(J[cols]).values
        print(f"{ds} {model}: judged {len(J)}; accepted by kind {J.groupby('kind').acc.mean().round(3).to_dict()}; spend ${H.spent:.3f}")
    elif mode == "judge":
        path, nuis, model = sys.argv[2], sys.argv[3], sys.argv[4]
        df = pd.read_csv(path).dropna(subset=["edited"])
        J = judge(df, nuis, model)
        J.to_csv(path.replace(".csv", f"_judged_{model.split('/')[-1]}.csv"), index=False)
        ok = J[J.ok]
        print(f"{nuis} {model}: judged {len(J)}; accepts {ok.accepted.mean():.0%} of auto-ok; spend ${H.spent:.3f} | "
              + " ".join(f"{c}={ok[c].astype(str).str[:3].value_counts().to_dict()}"
                         for c in ["target_same", "strength_same", "other_change", "edit_fluent"]))
