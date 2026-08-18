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

config → embeddings → attention → MLP → transformer block → full model → loss →
data pipeline → training → checkpointing → generation → pretrained weight loading →
reference validation
