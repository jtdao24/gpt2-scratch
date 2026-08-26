"""GPT-2 model components."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by "
                f"n_head ({config.n_head})"
            )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # One fused projection produces query, key, and value together.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # The output projection mixes the concatenated heads back together.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        # Stage 6 reads this marker to scale residual projections by
        # 1/sqrt(2*n_layer), keeping residual-stream variance in check.
        self.c_proj.GPT_SCALE_INIT = True

        mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )
        self.register_buffer(
            "bias",
            mask.view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(~self.bias[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """Position-wise feed-forward network used by GPT-2."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # Expand to four times the residual width.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        # GPT-2 uses the tanh approximation of GELU.
        self.gelu = nn.GELU(approximate="tanh")
        # Project back down to the residual width.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)

        # Stage 6 reads this marker to scale residual projections by
        # 1/sqrt(2*n_layer).
        self.c_proj.GPT_SCALE_INIT = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Pre-LayerNorm GPT-2 transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                # Token IDs map to learned token embeddings.
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Sequence positions map to learned positional embeddings.
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # The output projection reuses the token embedding table.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length T={T} exceeds block_size={self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
