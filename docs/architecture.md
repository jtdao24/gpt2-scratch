# Architecture and conventions

Reference notes for working on the model. The README covers what the project
does and how to use it; this file records the rules the code has to keep.

## Configuration

GPT-2 small, as defined by `GPTConfig()`:

| Field | Value | Meaning |
|---|---|---|
| `vocab_size` | 50,257 | 50,000 BPE merges, 256 byte tokens, and `<\|endoftext\|>` |
| `block_size` | 1,024 | maximum context length |
| `n_layer` | 12 | transformer blocks |
| `n_head` | 12 | attention heads per block |
| `n_embd` | 768 | residual stream width |
| `head_dim` | 64 | derived: `n_embd // n_head` |

`GPTConfig.from_pretrained_name` also covers `gpt2-medium`, `gpt2-large` and
`gpt2-xl`. Treat a config as immutable once built; it validates itself only at
construction, so derive changed configs with `dataclasses.replace`.

## Module names

Every parameter is named exactly as in Hugging Face's GPT-2, so OpenAI's weights
load by name with no remapping. All 149 state-dict keys match.

```text
transformer.wte, transformer.wpe
transformer.h.{i}.ln_1, .attn.c_attn, .attn.c_proj
transformer.h.{i}.ln_2, .mlp.c_fc,   .mlp.c_proj
transformer.ln_f
lm_head                              (no bias, weight tied to wte)
```

Renaming any of these breaks pretrained loading.

## Tensor shapes

`B` is batch size, `T` sequence length (`T <= block_size`), `C` is `n_embd`,
`nh` is `n_head`, and `hs` is `head_dim`.

```text
idx      (B, T)             int64, values in [0, vocab_size)
tok_emb  (B, T, C)
pos_emb  (T, C)             broadcast over the batch
x        (B, T, C)          residual stream, the same shape through every block
qkv      (B, T, 3C)         one c_attn projection, split into three
q, k, v  (B, nh, T, hs)     split with view(B, T, nh, hs).transpose(1, 2)
logits   (B, T, vocab_size)
loss     ()                 mean cross-entropy over all B*T positions
```

Attention uses `F.scaled_dot_product_attention(is_causal=True)`, which never
materializes the `(B, nh, T, T)` score matrix. The explicit version is kept in
`tests/test_attention.py` as a reference.

The model does no shifting of targets: `targets[b, t]` must be the token that
follows `idx[b, t]`. The data loader produces aligned pairs.

## Design points that GPT-2 compatibility depends on

- **Pre-LayerNorm.** Each block computes `x = x + attn(ln_1(x))` then
  `x = x + mlp(ln_2(x))`, normalizing inside the residual branch, and a final
  `ln_f` precedes the output layer.
- **Learned position embeddings**, which fix the context at 1,024 tokens.
- **GELU with the tanh approximation**, as in OpenAI's TensorFlow code. Exact
  GELU differs from it by up to 4.7 × 10⁻⁴ per activation.
- **Weight tying.** `lm_head.weight` is the same tensor as `transformer.wte.weight`.
  `named_parameters()` lists it once; `state_dict()` lists it under both names.
- **Initialization.** Weights use standard deviation 0.02, except the two
  residual output projections per block (`attn.c_proj`, `mlp.c_proj`), which use
  0.02/√(2 × n_layer). They are marked with a `GPT_SCALE_INIT` attribute.

## Loading pretrained weights

Hugging Face stores four weight matrices per block in TensorFlow's `Conv1D`
layout, `(in_features, out_features)`, which is the transpose of PyTorch's
`nn.Linear`:

```text
attn.c_attn.weight    (768, 2304)
attn.c_proj.weight    (768, 768)     square
mlp.c_fc.weight       (768, 3072)
mlp.c_proj.weight     (3072, 768)
```

They are transposed by name from an explicit list in `gpt/hf_weights.py`. Never
decide by comparing shapes: `attn.c_proj` is square, so a shape check cannot see
that it needs transposing. Skipping it produces a model that loads and runs
without error but whose logits differ from the reference by up to 121.
