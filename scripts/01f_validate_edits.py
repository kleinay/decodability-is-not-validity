"""Blind LLM-judge check of editor outputs (pilot csv from 01e): each (source, edited) pair is shown in random order to
gpt-oss-120b with the BiasBios judge prompt (incl. the first-name note). Reports acceptance and reasons.
  python scripts/01f_validate_edits.py data/bios/llm_edits_pilot_<tag>_seed<k>.csv"""
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dv.judging import accept  # noqa: E402
src = (ROOT / "scripts/02c_judge_edits.py").read_text()
ns = {}
exec(src[src.index("ATTR = {"):src.index("tok = AutoTokenizer")], ns)  # noqa: S102
KEY = (Path.home() / ".groq_key").read_text().strip()
JUDGE = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-oss-120b"
df = pd.read_csv(sys.argv[1]).dropna(subset=["edited"])
rng = np.random.RandomState(0)
rows = []
for _, r in df.iterrows():
    swap = rng.rand() < 0.5
    a, b = (r.edited, r.source) if swap else (r.source, r.edited)
    prompt = ns["PROMPT"].format(a=a[:1500], b=b[:1500], t=ns["ATTR"]["bios"][0], n=ns["ATTR"]["bios"][1], note=ns["NOTE"]["bios"])
    b = {"model": JUDGE, "temperature": 0, "max_tokens": 600, "response_format": {"type": "json_object"},
         "messages": [{"role": "user", "content": prompt}]}
    b.update({"reasoning_effort": "low", "include_reasoning": False} if "gpt-oss" in JUDGE else {"reasoning_effort": "none"})
    body = json.dumps(b).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "dv-judge/1.0"})
    for att in range(6):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]
            break
        except Exception:
            time.sleep(2 ** att); d = "{}"
    j = {k: str(v).lower() for k, v in json.loads(d).items()}
    rows.append(dict(uid=r.uid, auto_ok=r.ok, target_same=j.get("q1"), nuis_same=j.get("q2"), other_change=j.get("q5"),
                     src_fluent=j.get("q4" if swap else "q3"), edit_fluent=j.get("q3" if swap else "q4")))
J = pd.DataFrame(rows)
J["kind"] = "nuisance"
J["accepted"] = accept(J).values
print(f"{len(J)} edits | judge accepts {J.accepted.mean():.0%} | automatic checks pass {J.auto_ok.mean():.0%}")
print("judge verdict counts:", {c: J[c].value_counts().to_dict() for c in ["target_same", "nuis_same", "other_change"]})
print(pd.crosstab(J.auto_ok, J.accepted, rownames=["auto_ok"], colnames=["judge_accepted"]))
J.to_csv(sys.argv[1].replace(".csv", "_judged.csv"), index=False)
print("judge:", JUDGE)
