"""Candidate replacement for Civil: sentiment (T) x negation (N) on Rotten Tomatoes sentences. An LLM rewrites each
sentence so that it contains negation (if it had none) or none (if it had some), keeping its meaning and sentiment
("not good" <-> "bad", "never fails" <-> "always works") and every other word. Automatic checks flag outputs whose
negation status did not flip, or that change too much.

  python scripts/01h_negation_edits.py pilot <model> [n_per_cell] [seed] -> data/rt/llm_edits_pilot_<model>_seed<k>.csv
  python scripts/01h_negation_edits.py judge <csv> <judge_model>          -> <csv>_judged_<judge>.csv
Groq API; key read from ~/.groq_key (never printed); hard budget guard."""
import difflib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
KEY = (Path.home() / ".groq_key").read_text().strip()
PRICES = {"openai/gpt-oss-120b": (0.15e-6, 0.75e-6), "openai/gpt-oss-20b": (0.075e-6, 0.30e-6),
          "qwen/qwen3.8-27b": (0.30e-6, 0.90e-6)}   # USD/token, priced conservatively
BUDGET_USD = 2.00
spent, LOCK = 0.0, threading.Lock()

NEG = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|nowhere|cannot|without|hardly|barely)\b|n't\b")

COMMON = """Each replacement must read as natural, idiomatic English that a film critic could have written, keep the reviewer's opinion exactly as positive or negative and as strong as before (not weaker, stronger or mixed), and change nothing else in the sentence. The new phrase must fit grammatically where it stands: if a preceding article or verb must change too (e.g. "an unsatisfying ending" -> "a disappointing ending", not "an not satisfying ending"), include it in the original phrase. Copy each original phrase EXACTLY as it appears in the sentence (same lowercase and spacing, e.g. "is n't" if the sentence splits it). If no replacement satisfies all of this, return {{"skip": true}}.

Return only JSON: {{"replacements": [{{"original": "...", "new": "..."}}]}}

Sentence:
{text}"""

ADD = """We need a version of the movie-review sentence below that contains a negation (not, n't, no, never, nothing, without, ...) but means exactly the same thing.
Choose ONE short phrase and give a replacement for it that contains a negation and is an exact paraphrase. Good patterns: an adjective with its natural negated counterpart ("dull" -> "not interesting", "unfunny" -> "not funny", "forgettable" -> "not memorable"), a verb phrase ("fails to" -> "does not", "lacks" -> "has no", "always works" -> "never fails"), or a strength-preserving negated idiom ("a great film" -> "nothing short of a great film", "riveting" -> "never less than riveting"). Use only negated forms a critic would naturally write: avoid double negatives and litotes ("not unstylish", "not unpredictable", "not without"), and awkward antonyms ("not large", "not small", "not dispensable"). If no natural option exists, skip. Never just insert "not" in front of an unchanged phrase: that reverses its meaning.
""" + COMMON

REMOVE = """We need a version of the movie-review sentence below that contains NO negation at all (no not, n't, no, never, nothing, nobody, none, neither, nor, nowhere, cannot, without, hardly, barely) but means exactly the same thing.
Give a replacement for EVERY phrase that contains one of these words, each an exact un-negated paraphrase: e.g. "not good" -> "bad", "is n't funny" -> "is unfunny", "never fails" -> "always works", "without charm" -> "charmless", "nothing special" -> "unremarkable", "does n't work" -> "fails". Never just delete the negation word: that reverses the meaning.
""" + COMMON

JUDGE_T = "the reviewer's overall opinion of the movie (positive or negative)"
JUDGE_N = ("whether the text contains an explicit negation word (not, n't, no, never, nothing, none, nobody, nowhere, neither, "
           "nor, without, cannot, hardly, barely); words like 'fails', 'lacks' or 'un-' prefixes do NOT count")


def _post(model, body):
    global spent
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                          "User-Agent": "dv-edit/1.0"})
    pin, pout = PRICES[model]
    for attempt in range(8):
        if spent >= BUDGET_USD:
            return None
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            u = d.get("usage", {})
            with LOCK:
                spent += u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout
            return d["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("retry-after", 2 ** attempt)))
                continue
            print("HTTP", e.code, e.read()[:300], flush=True)
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def _body(model, prompt, max_tokens=1500, json_mode=False):
    b = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        b["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        b.update({"reasoning_effort": "low", "include_reasoning": False})
    if "qwen3" in model:
        b.update({"reasoning_effort": "none"})
    return b


def checks(src, out, n_src):
    if not out:
        return dict(ok=False, why="empty", frac_changed=np.nan)
    wa, wb = src.split(), out.split()
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    frac = changed / max(1, len(wa))
    why = []
    if bool(NEG.search(out)) == bool(n_src):
        why.append("negation not flipped")
    if frac > 0.35:
        why.append(f"changed {frac:.0%} of words")
    if abs(len(wb) - len(wa)) > max(4, 0.2 * len(wa)):
        why.append("length changed")
    return dict(ok=not why, why="; ".join(why) or "-", frac_changed=round(frac, 3))


def apply_replacements(text, out):
    """Apply the model's phrase replacements by exact match, so that only the named phrases change."""
    try:
        d = json.loads(out)
    except Exception:
        return None, "bad json"
    if d.get("skip"):
        return None, "skipped"
    reps = [r for r in d.get("replacements", []) if isinstance(r, dict) and r.get("original") and r.get("new") is not None]
    if not reps:
        return None, "no replacements"
    for r in reps:
        if r["original"] not in text:
            return None, "phrase not found"
        text = text.replace(r["original"], r["new"], 1)
    return text, "-"


def edit(model, src):
    def one(r):
        b = _body(model, (REMOVE if r.N else ADD).format(text=r.text), json_mode=True)
        if "gpt-oss" in model:
            b["reasoning_effort"] = "medium"
        out = _post(model, b)
        return apply_replacements(r.text, out) if out else (None, "empty")
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(one, [r for _, r in src.iterrows()]))
    rows = []
    for (_, r), (o, why) in zip(src.iterrows(), outs):
        c = checks(r.text, o, r.N) if o else dict(ok=False, why=why, frac_changed=np.nan)
        rows.append(dict(uid=r.uid, T=r["T"], N=r["N"], source=r.text, edited=o, **c))
    return pd.DataFrame(rows)


def judge(df, model):
    """Blind judge, same prompt and acceptance rule as the main pipeline (02c / src/dv/judging.py)."""
    from dv.judging import accept
    src = (ROOT / "scripts/02c_judge_edits.py").read_text()
    PROMPT = re.search(r'PROMPT = """(.*?)"""', src, re.S).group(1)
    rng = np.random.RandomState(0)
    swaps = rng.rand(len(df)) < 0.5

    def one(i):
        r = df.iloc[i]
        a, b = (r.edited, r.source) if swaps[i] else (r.source, r.edited)
        out = _post(model, _body(model, PROMPT.format(a=a, b=b, t=JUDGE_T, n=JUDGE_N, note=""), json_mode=True))
        try:
            return {k: str(v).lower() for k, v in json.loads(out or "{}").items()}
        except Exception:
            return {}
    with ThreadPoolExecutor(6) as ex:
        js = list(ex.map(one, range(len(df))))
    J = df.copy()
    J["target_same"] = [j.get("q1") for j in js]
    J["nuis_same"] = [j.get("q2") for j in js]
    J["other_change"] = [j.get("q5") for j in js]
    J["src_fluent"] = [j.get("q4" if s else "q3") for j, s in zip(js, swaps)]
    J["edit_fluent"] = [j.get("q3" if s else "q4") for j, s in zip(js, swaps)]
    J["kind"] = "nuisance"
    J["accepted"] = accept(J).values
    return J


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pilot":
        from datasets import load_dataset
        model = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 13   # 13: prompt development; another seed for validation
        t = load_dataset("cornell-movie-review-data/rotten_tomatoes")["test"].to_pandas()
        t["uid"] = "rt-test-" + t.index.astype(str)
        t["T"] = 1 - t.label          # target: negative sentiment (co-occurs with negation)
        t["N"] = t.text.map(lambda s: bool(NEG.search(s))).astype(int)
        src = t.groupby(["T", "N"]).sample(n=n, random_state=seed)
        df = edit(model, src)
        out = ROOT / f"data/rt/llm_edits_pilot_{model.split('/')[-1]}_seed{seed}.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"{model}: {len(df)} edits, automatic checks passed {df.ok.mean():.0%} "
              f"(add {df[df.N == 0].ok.mean():.0%}, remove {df[df.N == 1].ok.mean():.0%}); spend ${spent:.3f}")
        print(df.why.value_counts().head(8).to_string())
    elif mode == "judge":
        path, model = sys.argv[2], sys.argv[3]
        df = pd.read_csv(path).dropna(subset=["edited"])
        J = judge(df, model)
        J.to_csv(path.replace(".csv", f"_judged_{model.split('/')[-1]}.csv"), index=False)
        ok = J[J.ok]
        print(f"{model}: judged {len(J)}; accepted {J.accepted.mean():.0%} of all, {ok.accepted.mean():.0%} of auto-ok "
              f"(add {ok[ok.N == 0].accepted.mean():.0%}, remove {ok[ok.N == 1].accepted.mean():.0%}); spend ${spent:.3f}")
        for c in ["target_same", "nuis_same", "other_change", "edit_fluent"]:
            print(" ", c, ok[c].value_counts().to_dict())
