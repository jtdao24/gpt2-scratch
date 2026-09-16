"""Convert text to token IDs, store uint16 binaries, and produce batches."""

from pathlib import Path
from urllib import error, request

import numpy as np
import tiktoken
import torch


TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def download_text(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with request.urlopen(url) as response:
            content = response.read()
        if not content:
            raise RuntimeError(f"download from {url} produced an empty file")
        dest.write_bytes(content)
    except (error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"failed to download {url} to {dest}: {exc}") from exc

    if dest.stat().st_size == 0:
        raise RuntimeError(f"download from {url} produced an empty file at {dest}")
    return dest


def prepare(
    text: str,
    out_dir: Path,
    vocab_size: int = 50257,
    val_fraction: float = 0.1,
) -> dict[str, int]:
    ids = get_tokenizer().encode_ordinary(text)
    arr = np.array(ids, dtype=np.uint32)
    if arr.size == 0:
        raise ValueError("text must produce at least one token")

    max_id = int(arr.max())
    if max_id >= vocab_size:
        raise ValueError(
            f"maximum token id {max_id} must be less than vocab_size {vocab_size}"
        )
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"maximum token id {max_id} does not fit in uint16")

    n_val = int(len(arr) * val_fraction)
    val = arr[:n_val]
    train = arr[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train.astype(np.uint16).tofile(out_dir / "train.bin")
    val.astype(np.uint16).tofile(out_dir / "val.bin")

    return {"train": len(train), "val": len(val), "total": len(arr)}


def prepare_tinyshakespeare(
    data_dir: Path = Path("data"),
    url: str = TINYSHAKESPEARE_URL,
    vocab_size: int = 50257,
    val_fraction: float = 0.1,
) -> dict[str, int]:
    text_path = download_text(url, data_dir / "input.txt")
    text = text_path.read_text(encoding="utf-8")
    return prepare(
        text,
        data_dir,
        vocab_size=vocab_size,
        val_fraction=val_fraction,
    )


class DataLoader:
    def __init__(self, data_dir: Path, split: str, B: int, T: int) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        path = data_dir / f"{split}.bin"
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.B = B
        self.T = T
        required = B * T + 1
        available = len(self.tokens)
        if available < required:
            raise ValueError(
                f"{path} contains {available} tokens, but {required} are required"
            )
        self.position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        buf = torch.from_numpy(
            self.tokens[
                self.position : self.position + self.B * self.T + 1
            ].astype(np.int64)
        )
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.position += self.B * self.T
        if self.position + self.B * self.T + 1 > len(self.tokens):
            self.position = 0
        return x, y

    def reset(self) -> None:
        self.position = 0
