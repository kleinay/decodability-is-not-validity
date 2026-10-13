"""Mean-pooled residual-stream activations (special tokens excluded) at 7 depth fractions.
Usage: python scripts/02_extract.py <model_key> [dataset ...]  -> data/<ds>/acts_<model_key>.npz"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", {}),
    "gemma9b": ("google/gemma-2-9b-it", {"attn_implementation": "eager"}),
}
FRACS = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
MAX_LEN, BS = 256, 16

key = sys.argv[1]                      # e.g. "qwen7b" (mean pooling) or "qwen7b:last" (last non-special token)
datasets = sys.argv[2:] or ["cebab", "bios", "civil"]
key, _, pool = key.partition(":")
pool = pool or "mean"
name, kw = MODELS[key]
key = key if pool == "mean" else f"{key}_{pool}"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda", **kw).eval()
n_layers = model.config.num_hidden_layers
layers = sorted({max(1, round(f * n_layers)) for f in FRACS})
print(f"{name}: {n_layers} layers, extracting hidden_states {layers}", flush=True)

for ds in datasets:
    texts = pd.read_parquet(ROOT / "data" / ds / "texts.parquet")
    path = ROOT / "data" / ds / f"acts_{key}.npz"
    start, prev = 0, None
    if path.exists():  # rows appended to texts.parquet after a previous run: extract only the new tail
        prev = np.load(path, allow_pickle=True)
        start = len(prev["uid"])
        assert (prev["uid"] == texts.uid.values[:start]).all()
        if start == len(texts):
            print(f"{ds}: up to date", flush=True)
            continue
    tail = texts.iloc[start:]
    order = np.argsort(tail.text.str.len().values)
    out = np.zeros((len(tail), len(layers), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(order), BS):
            idx = order[i:i + BS]
            enc = tok(list(tail.text.values[idx]), return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN, return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask")
            enc = {k: v.cuda() for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True).hidden_states
            mask = (enc["attention_mask"].bool() & ~special.cuda().bool()).float().unsqueeze(-1)
            denom = mask.sum(1).clamp(min=1)
            if pool == "last":  # right padding: last position whose mask is 1
                last = (mask.squeeze(-1) * torch.arange(mask.shape[1], device=mask.device)).argmax(1)
            for j, l in enumerate(layers):
                if pool == "last":
                    out[idx, j] = hs[l][torch.arange(len(idx)), last].float().cpu().numpy()
                else:
                    out[idx, j] = ((hs[l].float() * mask).sum(1) / denom).cpu().numpy()
    assert np.isfinite(out).all()
    if prev is not None:
        assert (prev["layers"] == np.array(layers)).all()
        out = np.concatenate([prev["X"], out])
    np.savez(path, X=out, uid=texts.uid.values, layers=np.array(layers))
    print(f"{ds}: {out.shape} in {time.time()-t0:.0f}s", flush=True)
