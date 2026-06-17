#!/usr/bin/env python
"""Capture a real inference-stage KV cache from GPT-2 as a 4D tensor for the Julia demo.

This is the order-4 companion of extract_kv.py. Instead of collapsing the cache to one
(layer, head) and saving a 2D K/V matrix, we keep the full structure: we hook *every*
attention layer's combined QKV projection (`c_attn`), and for GPT-2 (no rotary
embeddings) those key/value projections are exactly what the KV cache stores after
prefill. Stacking over layers and heads yields the genuine cached keys/values as

    K, V : [n_layers, n_heads, seq, head_dim]      (4D)
    q    : [n_layers, n_heads, head_dim]           (query of the last position)

This layout maps directly onto LRDD's `Tucker4` modes (layer, head, token, feature),
so the Julia side can treat the whole cache as one 4D tensor.

Output: kv_cache_4d.npz, readable in Julia with `NPZ.npzread` (shape and indexing are
preserved: a numpy [L,H,S,D] array reads back as a Julia array of the same shape).
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

MODEL = "gpt2"        # 124M, 12 layers x 12 heads, head_dim = 64
MAX_TOKENS = 1024     # GPT-2 context limit
OUT = "kv_cache_4d.npz"

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

hidden = model.config.n_embd          # 768
n_head = model.config.n_head          # 12
n_layer = model.config.n_layer        # 12
head_dim = hidden // n_head           # 64

# Hook every layer's combined QKV projection: each output is [1, seq, 3*hidden].
captured = {}
handles = []
for li, block in enumerate(model.transformer.h):
    handles.append(block.attn.c_attn.register_forward_hook(
        lambda m, inp, out, li=li: captured.__setitem__(li, out.detach())
    ))
with torch.no_grad():
    model(ids, use_cache=True)
for h in handles:
    h.remove()

# Per layer, split the projection into q/k/v and reshape heads out:
#   [seq, hidden] -> [seq, n_head, head_dim] -> [n_head, seq, head_dim].
K = np.empty((n_layer, n_head, seq, head_dim), dtype=np.float32)
V = np.empty((n_layer, n_head, seq, head_dim), dtype=np.float32)
q = np.empty((n_layer, n_head, head_dim), dtype=np.float32)
for li in range(n_layer):
    q_all, k_all, v_all = captured[li][0].split(hidden, dim=-1)   # each [seq, hidden]
    heads = lambda x: x.reshape(seq, n_head, head_dim).permute(1, 0, 2).numpy()
    K[li] = heads(k_all)                 # [n_head, seq, head_dim] -- cached keys
    V[li] = heads(v_all)                 # [n_head, seq, head_dim] -- cached values
    q[li] = heads(q_all)[:, -1, :]       # [n_head, head_dim]      -- last-position query

np.savez(OUT, K=K, V=V, q=q)
print(f"saved {OUT}  K={K.shape} V={V.shape} q={q.shape}  "
      f"(model={MODEL}, n_layer={n_layer}, n_head={n_head}, seq={seq}, head_dim={head_dim})")
