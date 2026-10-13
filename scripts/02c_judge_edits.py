"""Blind LLM validation of every counterfactual / placebo pair (Qwen3-32B, thinking disabled; not one of the
probed models). The two texts are shown in random order, without saying which one is edited, and the judge
answers the same questions about target and nuisance, so it is not told which attribute is supposed to change.
Writes data/<ds>/judge.parquet (one row per pair); the human-annotation sample is 02f_human_audit_sample.py."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
JUDGE = "Qwen/Qwen3-32B"
BS = 32
ATTR = {
    "cebab": ("the reviewer's opinion of the SERVICE (positive, negative, or not mentioned)",
              "the reviewer's opinion of the FOOD (positive, negative, or not mentioned)"),
    "bios": ("the occupation of the person described", "the gender of the person described"),
    "civil": ("how toxic the comment is (rude, disrespectful, insulting or hateful language)",
              "which religious or sexual-orientation group, if any, the comment is about"),
}
PROMPT = """You will compare two short texts.

Text 1:
<<<{a}>>>

Text 2:
<<<{b}>>>

Answer each question about the two texts.
Q1. Is {t} the same in Text 1 and Text 2? (yes / no / unclear)
Q2. Is {n} the same in Text 1 and Text 2? (yes / no / unclear)
Q3. Is Text 1 fluent, grammatical English (as fluent as typical text of this kind)? (yes / no)
Q4. Is Text 2 fluent, grammatical English (as fluent as typical text of this kind)? (yes / no)
Q5. Apart from the attributes in Q1 and Q2, do the texts differ in meaning? (none / minor / major)

Reply with only a JSON object: {{"q1": ..., "q2": ..., "q3": ..., "q4": ..., "q5": ...}}"""

tok = AutoTokenizer.from_pretrained(JUDGE)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map="cuda").eval()
rng = np.random.RandomState(3)
for ds in sys.argv[1:] or ["cebab", "bios", "civil"]:
    tx = pd.read_parquet(ROOT / f"data/{ds}/texts.parquet").set_index("uid").text
    pr = pd.read_parquet(ROOT / f"data/{ds}/pairs.parquet").reset_index(drop=True)
    swap = rng.rand(len(pr)) < 0.5  # True: edited text shown as Text 1
    prompts = []
    for (_, p), sw in zip(pr.iterrows(), swap):
        a, b = tx[p.uid_a][:1500], tx[p.uid_b][:1500]
        if sw:
            a, b = b, a
        msg = [{"role": "user", "content": PROMPT.format(a=a, b=b, t=ATTR[ds][0], n=ATTR[ds][1])}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    order = np.argsort([len(p) for p in prompts])
    raw = [None] * len(prompts)
    for i in range(0, len(order), BS):
        idx = order[i:i + BS]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=80, do_sample=False)
        for j, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            raw[j] = tok.decode(g, skip_special_tokens=True)
        print(f"{ds}: {min(i + BS, len(order))}/{len(order)}", flush=True)

    def parse(s):
        m = re.search(r"\{.*\}", s, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        return {k: str(d.get(k, "")).strip().lower() for k in ["q1", "q2", "q3", "q4", "q5"]}

    P = pd.DataFrame([parse(r) for r in raw])
    out = pr.copy()
    out["raw"] = raw
    out["target_same"], out["nuis_same"], out["other_change"] = P.q1, P.q2, P.q5
    out["src_fluent"] = np.where(swap, P.q4, P.q3)
    out["edit_fluent"] = np.where(swap, P.q3, P.q4)
    out.to_parquet(ROOT / f"data/{ds}/judge.parquet")
    print(ds, out.groupby("kind")[["target_same", "nuis_same", "edit_fluent", "other_change"]]
          .agg(lambda x: x.value_counts(normalize=True).round(2).to_dict()).to_string(), flush=True)
