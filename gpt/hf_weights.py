"""Load pretrained tensors using Hugging Face only as a weight source.

Our GPT class remains the model implementation. Export a checkpoint, then use
the existing generation command unchanged:

    python -m gpt.hf_weights --model gpt2 --out out/gpt2_pretrained.pt
    python -m gpt.generate --checkpoint out/gpt2_pretrained.pt \
        --prompt "The meaning of life is" --tokens 40
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .config import GPTConfig
from .model import GPT

HF_MODEL_IDS: dict[str, str] = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
}

# attn.c_proj.weight is square: its Conv1D layout cannot be detected by shape.
# Always select these transposes by name, including the square projection.
TRANSPOSED_SUFFIXES: tuple[str, ...] = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def convert_state_dict(
    hf_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert named Conv1D weights to contiguous Linear weights."""
    return {
        key: value.t().contiguous()
        if key.endswith(TRANSPOSED_SUFFIXES)
        else value
        for key, value in hf_state.items()
    }


def from_pretrained(model_type: str = "gpt2", device: str = "cpu") -> GPT:
    """Strictly load Hugging Face weights into our GPT in evaluation mode."""
    if model_type not in HF_MODEL_IDS:
        valid_names = ", ".join(HF_MODEL_IDS)
        raise ValueError(
            f"unknown pretrained model name {model_type!r}; "
            f"valid names are: {valid_names}"
        )

    config = GPTConfig.from_pretrained_name(model_type)
    from transformers import GPT2LMHeadModel

    hf_state = GPT2LMHeadModel.from_pretrained(
        HF_MODEL_IDS[model_type]
    ).state_dict()
    model = GPT(config)
    model.load_state_dict(convert_state_dict(hf_state))
    model.to(device).eval()
    return model


def main(argv: list[str] | None = None) -> None:
    """Export pretrained weights in the project's existing checkpoint format."""
    parser = argparse.ArgumentParser(description="Export pretrained GPT-2 weights")
    parser.add_argument("--model", default="gpt2", choices=tuple(HF_MODEL_IDS))
    parser.add_argument(
        "--out", type=Path, default=Path("out/gpt2_pretrained.pt")
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    model = from_pretrained(args.model, device=args.device)
    config = model.config
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        args.out,
        model=model,
        optimizer=optimizer,
        step=0,
        train_position=0,
        val_loss=float("nan"),
        train_config={"source": "huggingface", "model_type": args.model},
        model_config=asdict(config),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Saved {args.out} ({parameter_count:,} parameters)")


if __name__ == "__main__":
    main()
