"""Stratified sample of edit pairs for blind human annotation -> data/human_audit_sample.csv (25 per dataset x edit
kind). Texts are shown in random order as text_1/text_2 (the key mapping to source/edit is in a separate file),
so annotators are blind to which text was edited."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.RandomState(11)
rows, key = [], []
for ds in ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid").text
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet")
    for kind, g in pr.groupby("kind"):
        for _, p in g.sample(n=min(25, len(g)), random_state=rng).iterrows():
            sw = rng.rand() < 0.5
            a, b = (p.uid_b, p.uid_a) if sw else (p.uid_a, p.uid_b)
            item = f"{ds}-{len(rows):04d}"
            rows.append(dict(item=item, dataset=ds, text_1=tx[a], text_2=tx[b], target_same="", nuisance_same="",
                             text_1_fluent="", text_2_fluent="", other_meaning_change="", notes=""))
            key.append(dict(item=item, kind=kind, uid_a=p.uid_a, uid_b=p.uid_b, edited_is_text_1=sw))
pd.DataFrame(rows).sample(frac=1, random_state=0).to_csv(ROOT / "data/human_audit_sample.csv", index=False)
pd.DataFrame(key).to_csv(ROOT / "data/human_audit_key.csv", index=False)
print(len(rows), "items")
