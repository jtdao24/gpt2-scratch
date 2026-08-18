"""Model-shape configuration for GPT-2, defaulting to GPT-2 124M (small)."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    def __post_init__(self) -> None:
        values = (
            ("block_size", self.block_size),
            ("vocab_size", self.vocab_size),
            ("n_layer", self.n_layer),
            ("n_head", self.n_head),
            ("n_embd", self.n_embd),
        )
        for field_name, value in values:
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by "
                f"n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_pretrained_name(cls, name: str) -> "GPTConfig":
        model_shapes: dict[str, tuple[int, int, int]] = {
            "gpt2": (12, 12, 768),
            "gpt2-medium": (24, 16, 1024),
            "gpt2-large": (36, 20, 1280),
            "gpt2-xl": (48, 25, 1600),
        }
        if name not in model_shapes:
            valid_names = ", ".join(model_shapes)
            raise ValueError(
                f"unknown pretrained model name {name!r}; "
                f"valid names are: {valid_names}"
            )

        n_layer, n_head, n_embd = model_shapes[name]
        return cls(
            block_size=1024,
            vocab_size=50257,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
