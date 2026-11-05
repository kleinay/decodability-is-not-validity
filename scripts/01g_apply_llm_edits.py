"""Replace BiasBios' rule-based gender-swap nuisance pairs with the LLM-hybrid edits (01e full).
 - existing 'nuisance' pairs are kept as kind 'nuisance_rule' (comparison only; never used for headline numbers)
 - accepted-by-automatic-checks edits are appended to the END of data/bios/texts.parquet (uid + '-llm', split 'cf')
   so cached activations stay row-aligned; pairs of kind 'nuisance' point to them. Idempotent."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
tag = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss-120b_hybrid"
tx = pd.read_parquet(ROOT / "data/bios/texts.parquet")
pr = pd.read_parquet(ROOT / "data/bios/pairs.parquet")
if tx.uid.str.endswith("-llm").any():
    sys.exit("already applied")
ed = pd.read_parquet(ROOT / f"data/bios/llm_edits_{tag}.parquet")
tx.to_parquet(ROOT / "data/bios/texts_before_llm_edits.parquet")
pr.to_parquet(ROOT / "data/bios/pairs_before_llm_edits.parquet")

src = tx.set_index("uid")
good = ed[ed.ok & ed.edited.notna() & (ed.edited != ed.source)].copy()
good = good[good.uid.isin(src.index[(src.split == "test")])]
new = pd.DataFrame({"uid": good.uid + "-llm", "text": good.edited.values, "split": "cf",
                    "T": good["T"].values, "N": 1.0 - good["N"].values})
pairs = pd.DataFrame({"uid_a": good.uid.values, "uid_b": new.uid.values, "kind": "nuisance",
                      "sign": [1 if n > o else -1 for n, o in zip(new.N, good["N"])]})
pr["kind"] = pr.kind.replace({"nuisance": "nuisance_rule"})
pd.concat([tx, new], ignore_index=True).to_parquet(ROOT / "data/bios/texts.parquet")
pd.concat([pr, pairs], ignore_index=True).to_parquet(ROOT / "data/bios/pairs.parquet")
print(f"edits: {len(ed)} generated, {len(good)} passed automatic checks -> {len(pairs)} 'nuisance' pairs "
      f"({int((pairs.sign == 1).sum())} F->M, {int((pairs.sign == -1).sum())} M->F); "
      f"{int((pr.kind == 'nuisance_rule').sum())} rule-based pairs kept as 'nuisance_rule'")
