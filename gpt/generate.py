"""Sample text from a trained GPT-2 model, with a command-line interface.

Example:

    python -m gpt.generate --checkpoint out/ckpt.pt --prompt "Hello" --tokens 50
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import load_checkpoint
from .config import GPTConfig
from .data import get_tokenizer
from .model import GPT


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = 50,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append autoregressively sampled tokens to a batch of tokenized prompts."""
    if temperature <= 0:
        raise ValueError(f"temperature must be greater than 0, got {temperature}")
    if max_new_tokens < 0:
        raise ValueError(
            f"max_new_tokens must be non-negative, got {max_new_tokens}"
        )
    if idx.ndim != 2:
        raise ValueError(f"idx must be 2-D, got shape {tuple(idx.shape)}")
    if idx.numel() == 0:
        raise ValueError("idx must be non-empty")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -model.config.block_size :]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k, dim=-1)
                    logits = logits.masked_fill(
                        logits < values[:, [-1]],
                        float("-inf"),
                    )
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                idx = torch.cat((idx, next_token), dim=1)
    finally:
        model.train(was_training)

    return idx


def generate_text(
    model: nn.Module,
    prompt: str,
    max_new_tokens: int,
    *,
    num_samples: int = 1,
    temperature: float = 1.0,
    top_k: int | None = 50,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> list[str]:
    """Tokenize a prompt, sample continuations, and decode each complete sequence."""
    tokenizer = get_tokenizer()
    prompt_ids = tokenizer.encode_ordinary(prompt)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")

    idx = torch.tensor(prompt_ids, dtype=torch.int64).unsqueeze(0)
    idx = idx.repeat(num_samples, 1).to(device)
    generated = generate(
        model,
        idx,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    return [tokenizer.decode(row.tolist()) for row in generated]


def load_model_from_checkpoint(path: Path, device: str = "cpu") -> GPT:
    """Rebuild a GPT model from a training checkpoint."""
    checkpoint = load_checkpoint(path, map_location="cpu")
    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config is not a dictionary")
    config = GPTConfig(**model_config)
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    del checkpoint
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained GPT-2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device)
        generator.manual_seed(args.seed)
    samples = generate_text(
        model,
        args.prompt,
        args.tokens,
        num_samples=args.samples,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        generator=generator,
    )
    for sample_index, sample in enumerate(samples):
        if sample_index > 0:
            print("\n" + "=" * 80 + "\n")
        print(sample)


if __name__ == "__main__":
    main()
