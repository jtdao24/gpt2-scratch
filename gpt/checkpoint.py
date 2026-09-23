"""Atomic GPT-2 training checkpoints made of tensors and safe-load primitives.

The checkpoint stores model and optimizer state, the number of completed optimizer
updates, the training-data cursor, the latest validation loss, and primitive config
dictionaries. It deliberately excludes live dataclass objects and ``Path`` values so
that PyTorch's default safe loader can read every checkpoint.
"""

import os
from pathlib import Path

import torch
from torch import nn


def _stringify_paths(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_stringify_paths(item) for item in value]
    return value


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_position: int,
    val_loss: float,
    train_config: dict[str, object],
    model_config: dict[str, object],
) -> None:
    """Atomically save state after ``step`` completed optimizer updates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "train_position": int(train_position),
        "val_loss": float(val_loss),
        "train_config": _stringify_paths(train_config),
        "model_config": _stringify_paths(model_config),
    }
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load a safe-format checkpoint onto ``map_location``."""
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return torch.load(path, map_location=map_location)
