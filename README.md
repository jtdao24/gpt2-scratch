# gpt2

GPT-2 (124M) implemented from scratch in PyTorch.

The model is built from PyTorch primitives. Hugging Face is used for two things
only: as a source of OpenAI's pretrained weights, and as a reference to check
the model against. It is never the model.

The project is built for correctness first. Components are tested against
independent reference implementations, not just for output shape, and the
finished model loads OpenAI's published weights and reproduces the reference
implementation's outputs.

## Current status

- A complete GPT-2 model: token and position embeddings, causal multi-head
  self-attention, a 4× MLP with tanh-approximated GELU, pre-LayerNorm blocks,
  and an output layer tied to the token embeddings. It has 124,439,808
  parameters, the same as OpenAI's GPT-2 small, and all 149 of its weight names
  match Hugging Face's, so pretrained weights load without renaming.
- Cross-entropy loss. An untrained model scores 10.998, close to the 10.825
  (ln 50,257) expected from a uniform guess over the vocabulary.
- A data pipeline: GPT-2 tokenization with tiktoken, token files stored as
  16-bit integers, and a batch loader that reads them without loading the whole
  file into memory.
- A training loop: AdamW with weight decay on weight matrices only, linear
  warmup into cosine decay, gradient accumulation, gradient clipping, bf16 mixed
  precision, periodic validation, and JSON Lines logs.
- Checkpointing of the model, optimizer state, step and data position, written
  atomically so an interrupted save cannot corrupt the previous checkpoint.
- Text generation with temperature and top-k sampling, from any checkpoint.
- Loading OpenAI's pretrained weights. The loaded model's logits match Hugging
  Face's implementation to within 4.6 × 10⁻⁵.

Not done yet: a validation harness comparing against the reference across many
inputs, and a long training run on a large corpus.

Naming rules, tensor shapes and the design points GPT-2 compatibility depends
on are in [docs/architecture.md](docs/architecture.md).

## Setup

Requires Python 3.11+.

```text
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPUs
pip install -r requirements.txt
```

Install PyTorch before the requirements, choosing the build for your hardware
at [pytorch.org](https://pytorch.org/get-started/locally/). On RTX 50-series
cards it must be built for CUDA 12.8 or newer: older builds install without
error but fail at the first GPU operation. Everything also runs on the CPU, and
the GPU-only tests are skipped there.

## Usage

### Generating text with OpenAI's weights

```text
$ python -m gpt.hf_weights --model gpt2 --out out/gpt2_pretrained.pt
Saved out\gpt2_pretrained.pt (124,439,808 parameters)

$ python -m gpt.generate --checkpoint out/gpt2_pretrained.pt --prompt "The meaning of life is" --tokens 40 --seed 42
The meaning of life is for you to discover it as you go along."

The author explains that she felt comfortable with those who would seek the freedom to say anything and for her to not be held under political influence.
```

The first command downloads OpenAI's weights from Hugging Face (about 500 MB,
cached after the first run) and saves them as a 475 MB checkpoint in this
project's format. The second loads that checkpoint into this project's model.

| Option | Meaning | Default |
|---|---|---|
| `--tokens` | number of tokens to generate | 100 |
| `--samples` | number of independent samples | 1 |
| `--temperature` | below 1 is more predictable, above 1 more varied | 1.0 |
| `--top-k` | sample only among the k most likely tokens; 0 turns this off | 50 |
| `--seed` | fixes the random choices so the output repeats | none |

On Windows, if characters such as dashes print as `�`, set `PYTHONUTF8=1`. The
generated text is correct; only the console display is affected.

### Training

Prepare the TinyShakespeare corpus, about 1 MB of text:

```text
$ python -c "from gpt.data import prepare_tinyshakespeare; print(prepare_tinyshakespeare())"
{'train': 304223, 'val': 33802, 'total': 338025}
```

Then train:

```text
$ python -m gpt.train --max-steps 50 --total-batch-size 32768 --warmup-steps 10 --eval-interval 10 --eval-steps 8
step     0 | loss 10.935944 | lr 6.0000e-05 | grad_norm 28.6464 | dt 1421.40 ms | tok/s 23053
step     1 | loss 9.661594 | lr 1.2000e-04 | grad_norm 9.2177 | dt 1291.25 ms | tok/s 25377
...
step    49 | loss 5.922194 | lr 6.0832e-05 | grad_norm 0.4837 | dt 1304.99 ms | tok/s 25110
```

Each line is one optimizer step. Over this run the validation loss fell from
10.93 to 6.14. The log is written to `out/log.jsonl`: one record per step, one
per validation, and a header record for each run that stores its settings.

The defaults follow GPT-2's training setup: 524,288 tokens per optimizer step
(128 accumulated batches of 4 × 1024 on an 8 GB card), a peak learning rate of
6 × 10⁻⁴ after 715 warmup steps, and 19,073 steps in total, or 10 billion
tokens. That needs a much larger corpus than TinyShakespeare.

A checkpoint is saved to `out/ckpt.pt` every `--checkpoint-interval` steps
(default 1000) and at the end of the run. `--resume` continues from it, and
refuses if the model or batch shape has changed. `--snapshot-interval N` also
keeps a separate checkpoint every N steps; at full size each is 1.4 GB.

On the CPU, a resumed run reproduces an uninterrupted one exactly. On a GPU,
training is not bit-for-bit reproducible even without interruption, and resumed
runs agree with uninterrupted ones to within that same variation.

`--compile` enables `torch.compile`. It needs Triton, so it does not currently
work on Windows.

## Tests

```text
python -m pytest -m "not network"   # offline, under 10 seconds
python -m pytest                    # everything; the first run downloads GPT-2 (about 500 MB)
```

There are 142 tests, 136 of which run offline. They check behavior rather than
only output shapes. Among them:

- a future token never changes the model's output at an earlier position;
- a transformer block becomes an exact identity when both of its branches are
  zeroed, which only holds if the residual connections are wired correctly;
- gradient accumulation produces the same gradients as one large batch;
- a resumed run matches an uninterrupted run exactly, on the CPU;
- a small model trained on a repeating pattern generates that pattern;
- the model loaded with OpenAI's weights matches Hugging Face's outputs, and a
  copy that skips one required transpose does not.

## Measured results

Measured during development on one machine: an RTX 5060 (8 GB), PyTorch
2.11.0+cu128, Python 3.12 and Windows 11. These are one-off measurements, not
results from a committed benchmark suite.

| Measurement | Result |
|---|---|
| Logits against Hugging Face, OpenAI's weights | max difference 4.6 × 10⁻⁵ |
| The same, with `attn.c_proj` left untransposed | max difference 121 |
| Training speed, batches of 4 × 1024, bf16 | about 25,000 tokens/s, 5.2 GB peak |
| The same with batches of 8 × 1024 | 8.8 GB, more than the card holds; about 6,000 tokens/s |
| One attention layer, forward and backward, 1 × 1024, fp32 | 43 MB with fused attention, about 220 MB computed explicitly |

At about 25,000 tokens/s, the full 10-billion-token training run would take
roughly 4.7 days on this card.

## Layout

```text
gpt/config.py            GPTConfig: model dimensions only
gpt/model.py             attention, MLP, transformer block, full model
gpt/data.py              tokenizer, corpus preparation, batch loader
gpt/train.py             training loop, learning-rate schedule, optimizer setup
gpt/checkpoint.py        atomic checkpoint saving and safe loading
gpt/generate.py          sampling, text generation, command-line tool
gpt/hf_weights.py        loads OpenAI's weights into the model
docs/architecture.md     naming rules, tensor shapes, compatibility notes
tests/                   test suite
```
