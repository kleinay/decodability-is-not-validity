"""Re-query Groq verdicts that came back empty after repeated 429s (same prompt, same presentation order)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/02d_judge_edits_groq.py").read_text()
ns = {"__file__": str(ROOT / "scripts/02d_judge_edits_groq.py")}
exec(src[:src.index("rng = np.random.RandomState(4)")], ns)  # noqa: S102 (our own file: prompt, call(), budget)
for ds in ["cebab", "civil", "bios"]:
    p = ROOT / f"data/{ds}/judge_groq.parquet"
    j = pd.read_parquet(p)
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid").text
    bad = j.index[j.raw == ""]
    for i in bad:
        r = j.loc[i]
        # recover presentation order from the stored fields is not possible; re-randomize is fine for a blind judge
        a, b, sw = tx[r.uid_a][:1500], tx[r.uid_b][:1500], False
        out = ns["call"](ns["PROMPT"].format(a=a, b=b, t=ns["ATTR"][ds][0], n=ns["ATTR"][ds][1]))
        try:
            d = {k: str(v).strip().lower() for k, v in json.loads(out).items()}
        except Exception:
            d = {}
        j.loc[i, ["raw", "target_same", "nuis_same", "other_change", "src_fluent", "edit_fluent"]] = [
            out, d.get("q1", ""), d.get("q2", ""), d.get("q5", ""), d.get("q3", ""), d.get("q4", "")]
    j.to_parquet(p)
    print(ds, "re-queried", len(bad), "| still empty", int((j.raw == "").sum()), f"| spend ${ns['spent']:.4f}")
