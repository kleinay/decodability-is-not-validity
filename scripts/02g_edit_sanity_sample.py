"""Sample of judged nuisance edits for a quick human sanity pass -> data/edit_sanity_sample.{html,csv}
For BiasBios gender swaps and both Civil identity-swap directions: N accepted and N rejected pairs (among pairs both
LLM judges rated), with word-level changes highlighted, both judges' answers, and the reason for rejection.
Contains dataset text (incl. toxic comments) -> gitignored, local use only."""
import difflib
import html
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from dv.judging import accept  # noqa: E402

N_PER = 8
rng = np.random.RandomState(5)
SETS = [("bios", "nuisance", "BiasBios: gender swap"),
        ("civil", "nuisance", "Civil: Muslim/LGBTQ → Christian/straight"),
        ("civil", "nuisance_rev", "Civil: Christian → Muslim")]
COLS = ["target_same", "nuis_same", "src_fluent", "edit_fluent", "other_change"]


def reason(j):
    y = lambda c: str(j[c]).startswith("yes")  # noqa: E731
    n = lambda c: str(j[c]).startswith("no")  # noqa: E731
    r = []
    if not y("target_same"):
        r.append("target changed/unclear")
    if not n("nuis_same"):
        r.append("nuisance not changed")
    if str(j.other_change).startswith("major"):
        r.append("major other change")
    if n("edit_fluent") and y("src_fluent"):
        r.append("fluency drop")
    return "; ".join(r) or "-"


def diff_html(a, b, window=260):
    wa, wb = a.split(" "), b.split(" ")
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    oa, ob, first = [], [], None
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        ta, tb = html.escape(" ".join(wa[i1:i2])), html.escape(" ".join(wb[j1:j2]))
        if op == "equal":
            oa.append(ta); ob.append(tb)
        else:
            first = len(" ".join(oa)) if first is None else first
            if ta:
                oa.append(f'<mark class="del">{ta}</mark>')
            if tb:
                ob.append(f'<mark class="ins">{tb}</mark>')
    sa, sb = " ".join(oa), " ".join(ob)
    if len(a) > 2 * window and first is not None:  # trim long texts around the first change
        lo = max(0, first - window)
        sa, sb = ("… " if lo else "") + sa[lo:lo + 3 * window] + " …", ("… " if lo else "") + sb[lo:lo + 3 * window] + " …"
    return sa, sb


rows = []
for ds, kind, label in SETS:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid")
    jq = pd.read_parquet(ROOT / f"data/{ds}/judge.parquet").set_index(["uid_a", "uid_b"])
    jg = pd.read_parquet(ROOT / f"data/{ds}/judge_groq.parquet").set_index(["uid_a", "uid_b"])
    both = jq.index.intersection(jg.index)
    jq, jg = jq.loc[both], jg.loc[both]
    sel = (jq.kind == kind)
    acc = accept(jq.reset_index()).values & accept(jg.reset_index()).values
    idx = np.where(sel.values)[0]
    for status, pool in [("accepted", idx[acc[idx]]), ("rejected", idx[~acc[idx]])]:
        for i in rng.choice(pool, min(N_PER, len(pool)), replace=False):
            ua, ub = both[i]
            a, b = tx.loc[ua, "text"], tx.loc[ub, "text"]
            qa, ga = jq.iloc[i], jg.iloc[i]
            rows.append(dict(set=label, status=status, toxic=int(tx.loc[ua, "T"]) if ds == "civil" else None,
                             occupation=("surgeon" if tx.loc[ua, "T"] == 1 else "nurse") if ds == "bios" else None,
                             original=a, edited=b,
                             qwen3_reason=reason(qa) if not accept(qa.to_frame().T).iloc[0] else "accept",
                             gptoss_reason=reason(ga) if not accept(ga.to_frame().T).iloc[0] else "accept",
                             **{f"qwen3_{c}": qa[c] for c in COLS}, **{f"gptoss_{c}": ga[c] for c in COLS},
                             your_verdict="", your_notes="", uid_a=ua, uid_b=ub))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data/edit_sanity_sample.csv", index=False)

# ------------------------------------------------------------------ HTML
css = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#ffffff;--acc:#1baf7a;--rej:#e34948;
--ins:#d7f2e6;--del:#fbe0df;--insInk:#0a5c3e;--delInk:#8c1f1e}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#33332f;--card:#222220;
--ins:#123d2c;--del:#4a1f1e;--insInk:#9be3c3;--delInk:#f3b3b2}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px}
.lead{color:var(--muted);max-width:900px;margin:0 0 12px}
.warn{border-left:3px solid var(--rej);padding:6px 10px;color:var(--muted);max-width:900px}
table{border-collapse:collapse;width:100%;max-width:1400px;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}
th{font-size:12px;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}
td.txt{width:32%;font-size:13px} td.meta{font-size:12px;color:var(--muted);width:12%}
mark.ins{background:var(--ins);color:var(--insInk);border-radius:3px;padding:0 2px}
mark.del{background:var(--del);color:var(--delInk);border-radius:3px;padding:0 2px;text-decoration:line-through}
.pill{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;font-weight:600;color:#fff}
.accepted{background:var(--acc)} .rejected{background:var(--rej)}
select,input{font:inherit;font-size:12px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:2px 4px;width:100%;box-sizing:border-box;margin-top:4px}
.wrap{overflow-x:auto}
"""
parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Edit sanity check</title><style>{css}</style></head><body>",
         "<h1>Counterfactual edits: sanity check</h1>",
         f"<p class=lead>{N_PER} edits accepted and {N_PER} rejected by the two LLM judges (Qwen3-32B, gpt-oss-120b) per edit type, "
         "random among pairs both judges rated. Removed words are struck through in red, inserted words in green. "
         "A pair is accepted only if <b>both</b> judges say the target is unchanged, the nuisance changed, no major other "
         "meaning change, and no fluency drop. Your verdicts are kept only in this page; the CSV next to this file has the same rows.</p>",
         "<p class=warn>Contains offensive comments from CivilComments.</p>"]
for label in df.set.unique():
    parts.append(f"<h2>{html.escape(label)}</h2><div class=wrap><table><tr><th>status</th><th>original</th><th>edited</th>"
                 "<th>judges</th><th>your call</th></tr>")
    for k, r in df[df.set == label].reset_index().iterrows():
        a, b = diff_html(r.original, r.edited)
        lab = (f"toxic={r.toxic}" if pd.notna(r.toxic) else f"{r.occupation}")
        parts.append(
            f"<tr><td class=meta><span class='pill {r.status}'>{r.status}</span><br>{html.escape(lab)}</td>"
            f"<td class=txt>{a}</td><td class=txt>{b}</td>"
            f"<td class=meta><b>Qwen3:</b> {html.escape(r.qwen3_reason)}<br><b>gpt-oss:</b> {html.escape(r.gptoss_reason)}</td>"
            f"<td class=meta><select><option></option><option>valid edit</option><option>target changed</option>"
            f"<option>meaning broken</option><option>nuisance not changed</option><option>other</option></select>"
            f"<input placeholder='notes'></td></tr>")
    parts.append("</table></div>")
parts.append("</body></html>")
(ROOT / "data/edit_sanity_sample.html").write_text("\n".join(parts))
print(df.groupby(["set", "status"]).size())
