"""GPT-2 model components."""

import torch
from torch import nn

from .config import GPTConfig


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
            )
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # Stage 6 extends this for nn.Linear and residual-projection scaling by
        # 1/sqrt(2*n_layer).
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length T={T} exceeds block_size={self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        return tok_emb + pos_emb
