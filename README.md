# gpt2

GPT-2 Small (124M) implemented from scratch in PyTorch, following Karpathy's GPT-2
reproduction workflow. The model is built from PyTorch primitives. Hugging Face is
used only for pretrained weight loading and reference validation.

## Architecture

```
vocab_size   50257
block_size    1024
n_layer         12
n_head          12
n_embd         768
```

### Conventions

Module names mirror Hugging Face GPT-2 exactly, so pretrained weights load without key
remapping in the pretrained-weight stage:

```
transformer.wte, transformer.wpe
transformer.h.{i}.ln_1, .attn.c_attn, .attn.c_proj
transformer.h.{i}.ln_2, .mlp.c_fc,   .mlp.c_proj
transformer.ln_f
lm_head                              (bias=False, weight tied to wte)
```

Shape contract, where `B` = batch, `T` = sequence length (`T <= block_size`),
`C` = `n_embd`, `nh` = `n_head`, `hs` = `head_dim` (= 64):

```
idx      (B, T)             int64, values in [0, vocab_size)
tok_emb  (B, T, C)
pos_emb  (T, C)             broadcast over batch
x        (B, T, C)          residual stream — invariant through every block
qkv      (B, T, 3C)         single c_attn projection, split into three
q, k, v  (B, nh, T, hs)
att      (B, nh, T, T)      causal-masked
logits   (B, T, vocab_size)
loss     ()                 scalar, mean cross-entropy over B*T
```

The residual stream stays `(B, T, C)` end to end.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install numpy tiktoken pytest
```

### GPU note

RTX 50-series (Blackwell, sm_120) requires a PyTorch/CUDA build with compatible
GPU architecture support. An incompatible build may install successfully but fail
when CUDA kernels execute.

Run tests with:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

## Plan

✅ config → ✅ embeddings → ✅ attention → ✅ MLP → ✅ transformer block → ✅ full model → ✅ loss →
✅ data pipeline → ✅ training → ✅ checkpointing → ✅ generation → pretrained weight loading →
reference validation
