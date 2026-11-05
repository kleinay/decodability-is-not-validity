"""Author reading of accepted Amazon edits: 50 pairs per setting (Amazon-Neg, Amazon-1P), accepted by both judges,
stratified by direction (25 add / 25 remove), shuffled together. Two raters read the same items independently
(the first author, and Claude as a second, non-human reader); judge verdicts are not shown.

Labels per item:  ok          same opinion and strength, natural, only the nuisance changed
                  unnatural   meaning kept, but stilted/awkward wording a reviewer would not write
                  wrong       opinion, strength or a fact changed, or the text is broken (beyond the source's own errors)

  python scripts/16_author_read_sample.py build     -> data/author_read/{sheet_author.html, sheet_claude.html, key.csv}
  python scripts/16_author_read_sample.py compare [a b] -> needs labels_<a>.csv, labels_<b>.csv (item,label,note), default author claude;
                                                       prints rates per setting and the agreement of the two raters"""
import difflib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dv.judging import accept  # noqa: E402

OUT = ROOT / "data/author_read"
INTENT = {("amzneg", 0): "add a negation", ("amzneg", 1): "remove every negation",
          ("amzfp", 0): "add first person (I/my/we)", ("amzfp", 1): "remove every first person"}


def accepted(ds):
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid")
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    pr = pr[pr.kind == "nuisance"].reset_index(drop=True)
    ok = np.ones(len(pr), bool)
    for f in ["judge.parquet", "judge_groq.parquet"]:
        j = pd.read_parquet(ROOT / f"data/{ds}/{f}")
        a = pd.Series(accept(j).values, index=pd.MultiIndex.from_frame(j[["uid_a", "uid_b"]]))
        ok &= a.reindex(pd.MultiIndex.from_frame(pr[["uid_a", "uid_b"]])).fillna(False).values.astype(bool)
    pr = pr[ok].copy()
    pr["src"], pr["edt"] = tx.text.reindex(pr.uid_a).values, tx.text.reindex(pr.uid_b).values
    pr["n_src"] = tx["N"].reindex(pr.uid_a).values.astype(int)
    return pr.assign(dataset=ds)


def diff_html(a, b):
    wa, wb = a.split(), b.split()
    sa, sb = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
        x, y = html.escape(" ".join(wa[i1:i2])), html.escape(" ".join(wb[j1:j2]))
        if op == "equal":
            sa.append(x), sb.append(y)
        else:
            if x:
                sa.append(f"<del>{x}</del>")
            if y:
                sb.append(f"<ins>{y}</ins>")
    return " ".join(sa), " ".join(sb)


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Edit reading sheet ({who})</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#222}}
.item{{border:1px solid #ccc;border-radius:6px;padding:.7em 1em;margin:1em 0}} .item.done{{border-color:#6a6;background:#f6fbf6}}
.meta{{color:#666;font-size:13px}} .t{{margin:.3em 0}} del{{background:#fdd;text-decoration:none}} ins{{background:#cfc;text-decoration:none}}
label{{margin-right:1.2em}} input[type=text]{{width:60%}} #bar{{position:sticky;top:0;background:#fff;padding:.5em 0;border-bottom:1px solid #ddd}}
</style></head><body><div id="bar"><b>Edit reading sheet ({who})</b> &mdash; <span id="cnt"></span>
<button onclick="dl()">Download labels CSV</button> <span class="meta">(saved automatically in this browser)</span></div>
<p><b>Task.</b> Each item shows a real Amazon review and an LLM edit meant to change only one surface feature
(negation, or the writer referring to themselves). Red = removed, green = inserted. Judge the <i>edited</i> text against the original:</p>
<ul><li><b>ok</b>: same opinion and same strength, reads naturally, only the intended feature changed.</li>
<li><b>unnatural</b>: meaning kept, but wording a reviewer would hardly write (stilted, odd register, awkward).</li>
<li><b>wrong</b>: opinion, its strength, or a fact changed; or the edit broke the text (beyond the original's own typos).</li></ul>
<p class="meta">Judge only what the edit changed; typos and informality already in the original are fine. Notes are optional.</p>
{items}
<script>
const K="dvread_{who}";let S=JSON.parse(localStorage.getItem(K)||"{{}}");
function upd(){{let n=0;document.querySelectorAll(".item").forEach(d=>{{const i=d.dataset.i,s=S[i]||{{}};
d.querySelectorAll("input[type=radio]").forEach(r=>r.checked=(r.value===s.label));d.querySelector("input[type=text]").value=s.note||"";
d.classList.toggle("done",!!s.label);if(s.label)n++;}});document.getElementById("cnt").textContent=n+" / {n} labelled";}}
document.addEventListener("change",e=>{{const d=e.target.closest(".item");if(!d)return;const i=d.dataset.i;S[i]=S[i]||{{}};
if(e.target.type==="radio")S[i].label=e.target.value;else S[i].note=e.target.value;localStorage.setItem(K,JSON.stringify(S));upd();}});
function dl(){{let r=["item,label,note"];document.querySelectorAll(".item").forEach(d=>{{const s=S[d.dataset.i]||{{}};
r.push(d.dataset.i+","+(s.label||"")+',"'+(s.note||"").replace(/"/g,'""')+'"');}});
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([r.join("\\n")],{{type:"text/csv"}}));a.download="labels_{who}.csv";a.click();}}
upd();</script></body></html>"""


def build():
    rng = np.random.RandomState(2026)
    parts = []
    for ds in ["amzneg", "amzfp"]:
        A = accepted(ds)
        for n_src, g in A.groupby("n_src"):
            parts.append(g.sample(n=25, random_state=rng))
    S = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
    S["item"] = np.arange(1, len(S) + 1)
    OUT.mkdir(exist_ok=True)
    S[["item", "dataset", "uid_a", "uid_b", "n_src", "src", "edt"]].to_csv(OUT / "key.csv", index=False)
    blocks = []
    for _, r in S.iterrows():
        a, b = diff_html(r.src, r.edt)
        blocks.append(f'<div class="item" data-i="{r["item"]}"><div class="meta">#{r["item"]} &middot; intended: '
                      f'{INTENT[(r.dataset, r.n_src)]}</div><div class="t"><b>original:</b> {a}</div>'
                      f'<div class="t"><b>edited:</b> {b}</div><div>'
                      + "".join(f'<label><input type="radio" name="l{r["item"]}" value="{v}"> {v}</label>'
                                for v in ["ok", "unnatural", "wrong"])
                      + ' <input type="text" placeholder="note"></div></div>')
    for who in ["author", "claude"]:
        (OUT / f"sheet_{who}.html").write_text(PAGE.format(who=who, items="\n".join(blocks), n=len(S)))
    print(f"{len(S)} items:", S.groupby(["dataset", "n_src"]).size().to_dict())


def compare(a="author", b="claude"):
    from sklearn.metrics import cohen_kappa_score
    K = pd.read_csv(OUT / "key.csv")
    for w in (a, b):
        K[w] = K["item"].map(pd.read_csv(OUT / f"labels_{w}.csv").set_index("item").label)
    for w in (a, b):
        print(w, K.groupby("dataset")[w].value_counts(normalize=True).round(2).unstack().to_string(), sep="\n")
    both = K.dropna(subset=[a, b])
    wa, wb = both[a] == "wrong", both[b] == "wrong"
    print(f"n={len(both)}  exact agreement {np.mean(both[a] == both[b]):.2f}, kappa {cohen_kappa_score(both[a], both[b]):.2f}; "
          f"wrong vs not: agreement {np.mean(wa == wb):.2f}, kappa {cohen_kappa_score(wa, wb):.2f}")
    for ds, g in both.groupby("dataset"):
        print(f"  {ds}: exact {np.mean(g[a] == g[b]):.2f}, kappa {cohen_kappa_score(g[a], g[b]):.2f}")
    print(pd.crosstab(both[a], both[b]).to_string())
    json.dump({w: K.groupby("dataset")[w].value_counts(normalize=True).round(3).unstack().to_dict() for w in (a, b)},
              open(OUT / f"rates_{a}_{b}.json", "w"), indent=1)
    return both


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else compare(*sys.argv[2:4])
