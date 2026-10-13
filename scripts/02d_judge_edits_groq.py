"""Second, independent LLM judge for the edit pairs, via the Groq API (gpt-oss-120b: a different family from the local
judge, Qwen3-32B, and from both probed models; Llama-3.3-70B was no longer served). Same blind prompt and random order as
02c_judge_edits.py. Hard budget guard: stops once estimated spend reaches BUDGET_USD.
Key is read from ~/.groq_key (never printed). Writes data/<ds>/judge_groq.parquet."""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL = "openai/gpt-oss-120b"
PRICE_IN, PRICE_OUT = 0.15e-6, 0.75e-6       # USD per token; output priced conservatively (reasoning tokens are billed)
BUDGET_USD = 1.60                             # user allowed up to $2
KEY = (Path.home() / ".groq_key").read_text().strip()

# share the exact prompt and attribute wording with the local judge
src = (ROOT / "scripts/02c_judge_edits.py").read_text()
ns = {}
exec(src[src.index("ATTR = {"):src.index("tok = AutoTokenizer")], ns)  # noqa: S102 (our own file)
ATTR, PROMPT = ns["ATTR"], ns["PROMPT"]

# all CEBaB and Civil pairs; a stratified sample of BiasBios keeps the total well inside the budget
WORKERS = 6
CAP = {"bios": {"nuisance": 800, "placebo": 400}}
spent = 0.0
n429 = 0
LOCK = threading.Lock()


def call(prompt):
    global spent, n429
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 600, "reasoning_effort": "low",
                       "include_reasoning": False, "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                          "User-Agent": "dv-judge/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            u = d.get("usage", {})
            with LOCK:
                spent += u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                with LOCK:
                    n429 += 1
                time.sleep(float(e.headers.get("retry-after", 2 ** attempt)))
                continue
            print("HTTP", e.code, e.read()[:200], flush=True)
            return ""
        except Exception:
            time.sleep(2 ** attempt)
    return ""


rng = np.random.RandomState(4)  # presentation order randomized independently of the local judge
for ds in sys.argv[1:] or ["cebab", "civil", "bios"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid").text
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet").reset_index(drop=True)
    swap = rng.rand(len(pr)) < 0.5
    pr["swap"] = swap
    keep = []
    for kind, g in pr.groupby("kind"):
        cap = CAP.get(ds, {}).get(kind)
        keep.append(g if cap is None or len(g) <= cap else g.sample(n=cap, random_state=0))
    sub = pd.concat(keep).sort_index()
    def judge(p):
        if spent >= BUDGET_USD:
            return None
        a, b = tx[p.uid_a][:1500], tx[p.uid_b][:1500]
        if p.swap:
            a, b = b, a
        out = call(PROMPT.format(a=a, b=b, t=ATTR[ds][0], n=ATTR[ds][1]))
        try:
            d = {k: str(v).strip().lower() for k, v in json.loads(out).items()}
        except Exception:
            d = {}
        return dict(uid_a=p.uid_a, uid_b=p.uid_b, kind=p.kind, raw=out,
                    target_same=d.get("q1", ""), nuis_same=d.get("q2", ""), other_change=d.get("q5", ""),
                    src_fluent=d.get("q4" if p.swap else "q3", ""), edit_fluent=d.get("q3" if p.swap else "q4", ""))

    rows = []
    with ThreadPoolExecutor(WORKERS) as ex:
        for r in ex.map(judge, [p for _, p in sub.iterrows()]):
            if r is not None:
                rows.append(r)
                if len(rows) % 200 == 0:
                    print(f"{ds}: {len(rows)}/{len(sub)}  spent ${spent:.3f}  429s {n429}", flush=True)
    if spent >= BUDGET_USD:
        print(f"budget reached (${spent:.2f})", flush=True)
    pd.DataFrame(rows).to_parquet(ROOT / f"data/{ds}/judge_groq.parquet")
    print(f"{ds}: done {len(rows)} pairs; cumulative spend ${spent:.3f}", flush=True)
