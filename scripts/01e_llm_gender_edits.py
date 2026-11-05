"""LLM-based gender edits for BiasBios (replacing the rule-based pronoun swap, which left gendered first names
unchanged). An LLM rewrites each test biography so that its subject has the other gender -- pronouns, gendered
nouns/titles and the subject's first name -- and leaves every other word unchanged. Automatic checks flag outputs
whose pronouns did not flip, that keep source-gender pronouns, change too much, or drop the occupation.

  python scripts/01e_llm_gender_edits.py pilot <model> [n]   -> data/bios/llm_edits_pilot_<model>.csv (no data changes)
  python scripts/01e_llm_gender_edits.py full <model>        -> data/bios/llm_edits_<model>.parquet
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
KEY = (Path.home() / ".groq_key").read_text().strip()
PRICES = {"openai/gpt-oss-20b": (0.075e-6, 0.30e-6), "openai/gpt-oss-120b": (0.15e-6, 0.75e-6), "qwen/qwen3.8-27b": (0.30e-6, 0.90e-6),
          "qwen/qwen3.6-27b": (0.30e-6, 0.90e-6)}   # USD/token, priced conservatively
BUDGET_USD = 3.00
spent, LOCK = 0.0, threading.Lock()

INSTRUCTION = """Rewrite the biography below so that the person it is about (its subject) has the other gender (male <-> female).

Step 1. Identify the subject: the person the biography describes, usually the one referred to as he/she.
Step 2. Change every word that refers to the subject:
- pronouns (he/she, him/her, his/her, himself/herself);
- gendered nouns and titles for the subject (e.g. Mr./Ms./Mrs., father/mother, son/daughter, businessman/businesswoman);
- the subject's FIRST NAME, wherever it appears (also inside the full name): replace it with the closest first name of the other gender from the same language and culture, preferably a direct counterpart (e.g. Maria -> Mario, Andrea -> Andrew, Antonello -> Antonella, Michael -> Michelle, Rhoda -> Robert), and use that new name consistently. If the first name is unisex (e.g. Chris, Alex, Jordan), keep it.
Step 3. Change nothing else. Other people mentioned in the text keep their names, pronouns and gender. Keep surnames, places, organizations, occupation, facts, wording, spelling (even typos), punctuation and length exactly as they are. Do not add or remove information.
If the text contains no reference to the subject's gender, return it unchanged.

Return only the rewritten biography, with no explanation.

Biography:
{text}"""

MALE = r"\b(he|him|his|himself|mr)\b"
FEMALE = r"\b(she|her|hers|herself|mrs|ms)\b"
OCC = r"\b(surgeon|surgery|surgical|nurse|nursing|rn|np)\b"
GENDERED = set("he she him her his hers himself herself mr mr. mrs mrs. ms ms. man woman men women father mother son "
               "daughter husband wife boyfriend girlfriend brother sister dad mom male female businessman businesswoman "
               "chairman chairwoman spokesman spokeswoman sir madam".split())


try:
    import gender_guesser.detector as _gg
    _DET = _gg.Detector(case_sensitive=False)
except ImportError:  # pip install gender-guesser
    _DET = None
_STOP = {"will", "mark", "may", "grace", "major", "bill", "frank", "rich", "hope", "art", "pat", "dean", "rose", "faith"}


_TITLES = {"dr", "mr", "ms", "mrs", "prof", "professor", "miss", "sir"}


def _names_of_gender(text, male):
    """First tokens of capitalized runs (so surnames and places like 'Sarah Wyatt', 'New York' are skipped) that a
    name lexicon classifies as (mostly) male / female first names."""
    want = {"male", "mostly_male"} if male else {"female", "mostly_female"}
    toks = re.findall(r"[A-Za-z][A-Za-z.'-]*", text)
    out = []
    for i, w in enumerate(toks):
        core = w.strip(".'-")
        if not (core[:1].isupper() and core[1:].islower() and len(core) > 2) or core.lower() in _STOP:
            continue
        prev = toks[i - 1].strip(".'-") if i else ""
        if prev[:1].isupper() and prev.lower() not in _TITLES:  # second/third token of a capitalized run
            continue
        if _DET.get_gender(core) in want:
            out.append(core)
    return out


def _norm(w):
    return re.sub(r"[^\w.]", "", w.lower()).strip(".") if w not in ("Mr.", "Ms.", "Mrs.") else w.lower()


def call(model, text):
    body = {"model": model, "temperature": 0, "max_tokens": 1500,
            "messages": [{"role": "user", "content": INSTRUCTION.format(text=text)}]}
    if "gpt-oss" in model:
        body.update({"reasoning_effort": "low", "include_reasoning": False})
    if "qwen3" in model:
        body.update({"reasoning_effort": "none"})
    return _post(model, body)


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


NAME_PROMPT = """Below is a biography of one person (the subject), who is {g_src}. The replacement names must be {g_dst} first names.

List the subject's FIRST NAMES exactly as they appear in the text (a first name may appear several times, alone or inside the full name), and give for each a replacement first name of the other gender: the closest direct counterpart if one exists (Maria -> Mario, Andrea -> Andrew, Antonello -> Antonella, Michael -> Michelle, Rhoda -> Robert), otherwise a common {g_dst} name of the same language and culture. Use a real, common name; do not just add a suffix.
A first name often appears after a title or with a middle initial: in "Mr. Timothy J. Mannen" the first name is "Timothy"; in "Dr. Shuja attended ..." the subject is referred to by the given name "Shuja". Look at the whole text, including the middle and the end, not only the beginning.
Do NOT include: surnames (e.g. "Mannen"); names of other people; places or organizations. If the subject is referred to only by a surname (e.g. "Dr. Tan" where Tan is a surname) or no first name appears, return an empty object.

Return only JSON: {{"replacements": {{"OldName": "NewName"}}}}

Biography:
{text}"""


def hybrid_edit(model, text, n_src):
    """LLM chooses the name replacements; the rule-based swap (src/dv/data.py) flips pronouns, titles and kinship
    nouns; names are substituted whole-word. Returns (edited_text or None, replacements dict)."""
    sys.path.insert(0, str(ROOT / "src"))
    from dv.data import gender_swap
    prompt = NAME_PROMPT.format(g_src="male" if n_src else "female", g_dst="female" if n_src else "male", text=text)
    body = {"model": model, "temperature": 0, "max_tokens": 300, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}
    if "qwen3" in model:
        body.update({"reasoning_effort": "none"})
    if "gpt-oss" in model:
        body.update({"reasoning_effort": "low", "include_reasoning": False, "max_tokens": 700})
    out = _post(model, body)
    if out is None:
        return None, {}
    try:
        rep = json.loads(out).get("replacements", {})
        rep = {k: v for k, v in rep.items() if isinstance(k, str) and isinstance(v, str) and k != v and k in text}
    except Exception:
        return None, {}
    edited = gender_swap(text)
    for old, new in sorted(rep.items(), key=lambda kv: -len(kv[0])):
        edited = re.sub(r"(?<![A-Za-z])" + re.escape(old) + r"(?![A-Za-z])", new, edited)
    return edited, rep


def checks(src, out, n_src):
    """n_src: 1 if the source subject is male. Returns dict of automatic checks."""
    if not out:
        return dict(ok=False, why="empty")
    s, o = src.lower(), out.lower()
    sm, sf = len(re.findall(MALE, s)), len(re.findall(FEMALE, s))
    om, of = len(re.findall(MALE, o)), len(re.findall(FEMALE, o))
    wa, wb = re.findall(r"\w+|[^\w\s]", src), re.findall(r"\w+|[^\w\s]", out)
    sm_ = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm_.get_opcodes() if op != "equal")
    frac = changed / max(1, len(wa))
    occ_kept = sorted(re.findall(OCC, s)) == sorted(re.findall(OCC, o))
    # every changed source word must be gendered or a capitalized name; pure insertions are not allowed
    bad_tokens, inserted = [], []
    for op, i1, i2, j1, j2 in sm_.get_opcodes():
        if op == "insert":
            inserted += wb[j1:j2]
        elif op in ("replace", "delete"):
            for w in wa[i1:i2]:
                core = re.sub(r"[^\w]", "", w)
                if _norm(w) not in GENDERED and not (core[:1].isupper() and core[1:].islower() or core.replace(".", "").isalpha() and core[:1].isupper()):
                    bad_tokens.append(w)
    had = (sm + sf) > 0
    if n_src == 1:
        flipped, leftover = (of > 0) if had else True, om
    else:
        flipped, leftover = (om > 0) if had else True, of
    why = []
    name_flag = ""
    if _DET is not None:  # ADVISORY ONLY: lexicon heuristics confuse surnames/organizations with first names, so this
        ns_, no_ = _names_of_gender(src, bool(n_src)), _names_of_gender(out, bool(n_src))  # is not part of `ok`
        if ns_ and sorted(no_) == sorted(ns_):
            name_flag = ns_[0]
    if not flipped:
        why.append("pronouns not flipped")
    if leftover:
        why.append(f"{leftover} source-gender pronoun(s) left")
    if frac > 0.25:
        why.append(f"changed {frac:.0%} of words")
    if not occ_kept:
        why.append("occupation terms changed")
    if inserted:
        why.append("inserted: " + " ".join(inserted[:4]))
    if bad_tokens:
        why.append("changed non-gendered word(s): " + " ".join(bad_tokens[:4]))
    if abs(len(wb) - len(wa)) > max(5, 0.1 * len(wa)):
        why.append("length changed")
    return dict(ok=not why, why="; ".join(why) or "-", frac_changed=round(frac, 3), changed_words=changed,
                name_flag=name_flag)


def run(model, src, hybrid=False):
    def one(r):
        if hybrid:
            return hybrid_edit(model, r.text, int(r["N"]))
        return call(model, r.text), {}
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(one, [r for _, r in src.iterrows()]))
    rows = []
    for (_, r), (o, rep) in zip(src.iterrows(), outs):
        c = checks(r.text, o, int(r["N"]))
        if hybrid and o and _DET is not None:   # replacement names must look like first names of the target gender
            want = {"female", "mostly_female", "andy"} if int(r["N"]) else {"male", "mostly_male", "andy"}
            odd = [v for v in rep.values() if _DET.get_gender(v) not in want | {"unknown"}]
            if odd:
                c["ok"], c["why"] = False, f"replacement name(s) of wrong gender: {odd}"
            unk = [v for v in rep.values() if _DET.get_gender(v) == "unknown"]
            if unk:
                c["name_flag"] = (c.get("name_flag", "") + " unknown-name:" + ",".join(unk)).strip()
        rows.append(dict(uid=r.uid, T=r["T"], N=r["N"], source=r.text, edited=o, replacements=json.dumps(rep), **c))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    hybrid = "--hybrid" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--hybrid"]
    mode, model = sys.argv[1], sys.argv[2]
    tx = pd.read_parquet(ROOT / "data/bios/texts.parquet")
    test = tx[(tx.split == "test") & tx["T"].notna() & tx["N"].notna()]
    tag = model.split("/")[-1]
    if mode == "pilot":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 13   # 13: prompt-development sample; use another for validation
        src = test.sample(n=n, random_state=seed)
        df = run(model, src, hybrid)
        df.to_csv(ROOT / f"data/bios/llm_edits_pilot_{tag}{'_hybrid' if hybrid else ''}_seed{seed}.csv", index=False)
    else:
        df = run(model, test, hybrid)
        df.to_parquet(ROOT / f"data/bios/llm_edits_{tag}{'_hybrid' if hybrid else ''}.parquet")
    print(f"{model}: {len(df)} edits, automatic checks passed {df.ok.mean():.0%}; spend ${spent:.3f}")
    print(df.why.value_counts().head(8).to_string())
