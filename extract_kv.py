#!/usr/bin/env python
"""Capture a real inference-stage KV cache from GPT-2 and save it for the Julia demo.

We run GPT-2 over a ~1k-token WikiText-103 passage with use_cache=True and grab the
key/value projections of one (layer, head) via a forward hook on `c_attn`. For GPT-2
(no rotary embeddings) those projections are exactly what the KV cache stores after
prefill, so K/V here are the genuine cached keys/values. We also grab the query of the
last position -- the vector a decode step would attend to all cached keys with.

Output: kv_cache.npz with K (seq x head_dim), V (seq x head_dim), q (head_dim).
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

MODEL = "gpt2"        # 124M, 12 layers x 12 heads, head_dim = 64
LAYER = 6             # mid-stack attention layer
HEAD = None           # None = all heads concatenated (d = hidden = 768); int = one head (d = 64)
MAX_TOKENS = 1024     # GPT-2 context limit

tok = GPT2TokenizerFast.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL).eval()

# Build a ~MAX_TOKENS passage by concatenating WikiText-103 lines (streamed, no full DL).
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
buf = ""
for row in ds:
    t = row["text"].strip()
    if t:
        buf += t + " "
    if len(tok(buf)["input_ids"]) >= MAX_TOKENS:
        break
ids = tok(buf, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"]
seq = ids.shape[1]

# Hook the chosen layer's combined QKV projection: output is [1, seq, 3*hidden].
captured = {}
handle = model.transformer.h[LAYER].attn.c_attn.register_forward_hook(
    lambda m, inp, out: captured.__setitem__("qkv", out.detach())
)
with torch.no_grad():
    model(ids, use_cache=True)
handle.remove()

hidden = model.config.n_embd          # 768
n_head = model.config.n_head          # 12
head_dim = hidden // n_head           # 64

q_all, k_all, v_all = captured["qkv"][0].split(hidden, dim=-1)   # each [seq, hidden]
if HEAD is None:                      # all heads concatenated -> d = hidden
    sel = lambda x: x.numpy()
else:                                 # one head -> d = head_dim
    sel = lambda x: x.reshape(seq, n_head, head_dim)[:, HEAD, :].numpy()

K = sel(k_all)                        # seq x d  -- cached keys
V = sel(v_all)                        # seq x d  -- cached values
q = sel(q_all)[-1]                    # d        -- query of the last position

np.savez("kv_cache.npz", K=K, V=V, q=q)
print(f"saved kv_cache.npz  K={K.shape} V={V.shape} q={q.shape}  "
      f"(model={MODEL}, layer={LAYER}, head={HEAD}, seq={seq})")
