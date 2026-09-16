from pathlib import Path

import numpy as np
import pytest
import torch

from gpt import GPT, GPTConfig
from gpt.data import (
    TINYSHAKESPEARE_URL,
    DataLoader,
    download_text,
    get_tokenizer,
    prepare,
)


SAMPLE_TEXT = "To be, or not to be, that is the question.\n" * 20


def write_tokens(data_dir: Path, split: str, count: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.arange(count, dtype=np.uint16).tofile(data_dir / f"{split}.bin")


def test_tokenizer_roundtrip() -> None:
    tokenizer = get_tokenizer()
    text = "Hello, GPT-2! 🎭\n"

    ids = tokenizer.encode_ordinary(text)

    assert tokenizer.decode(ids) == text
    assert tokenizer.n_vocab == 50257


def test_prepare_writes_uint16_sized_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "prepared"

    result = prepare(SAMPLE_TEXT, out_dir)

    assert (out_dir / "train.bin").stat().st_size == 2 * result["train"]
    assert (out_dir / "val.bin").stat().st_size == 2 * result["val"]


def test_prepared_files_contain_valid_uint16_ids(tmp_path: Path) -> None:
    out_dir = tmp_path / "prepared"
    prepare(SAMPLE_TEXT, out_dir)

    train = np.fromfile(out_dir / "train.bin", dtype=np.uint16)
    val = np.fromfile(out_dir / "val.bin", dtype=np.uint16)

    assert train.dtype == np.uint16
    assert val.dtype == np.uint16
    assert np.all(train < 50257)
    assert np.all(val < 50257)


def test_prepare_split_arithmetic(tmp_path: Path) -> None:
    val_fraction = 0.2

    result = prepare(SAMPLE_TEXT, tmp_path / "prepared", val_fraction=val_fraction)

    assert result["train"] + result["val"] == result["total"]
    assert result["val"] == int(result["total"] * val_fraction)


def test_prepare_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prepare("", tmp_path / "prepared")


def test_prepare_rejects_token_outside_vocabulary(tmp_path: Path) -> None:
    text = "Hello, world!"
    max_id = max(get_tokenizer().encode_ordinary(text))

    with pytest.raises(ValueError, match=str(max_id)):
        prepare(text, tmp_path / "prepared", vocab_size=50)


def test_batch_alignment(tmp_path: Path) -> None:
    write_tokens(tmp_path, "train", 100)
    loader = DataLoader(tmp_path, "train", B=2, T=4)

    x, y = loader.next_batch()

    assert torch.equal(x[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y[0], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(x[1], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(y[1], torch.tensor([5, 6, 7, 8]))

    x2, y2 = loader.next_batch()
    assert torch.equal(y2[:, :-1], x2[:, 1:])


def test_batch_shapes_and_dtype(tmp_path: Path) -> None:
    write_tokens(tmp_path, "train", 100)
    loader = DataLoader(tmp_path, "train", B=2, T=4)

    x, y = loader.next_batch()

    assert x.shape == y.shape == (2, 4)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_cursor_advances_by_batch_size(tmp_path: Path) -> None:
    write_tokens(tmp_path, "train", 100)
    loader = DataLoader(tmp_path, "train", B=2, T=4)

    x1, _ = loader.next_batch()
    x2, _ = loader.next_batch()

    assert not torch.equal(x1, x2)
    assert x2.flatten()[0].item() == 8


def test_cursor_wraps_to_start(tmp_path: Path) -> None:
    write_tokens(tmp_path, "train", 17)
    loader = DataLoader(tmp_path, "train", B=2, T=4)

    first = loader.next_batch()
    loader.next_batch()
    wrapped = loader.next_batch()

    assert torch.equal(wrapped[0], first[0])
    assert torch.equal(wrapped[1], first[1])


def test_reset_returns_to_start(tmp_path: Path) -> None:
    write_tokens(tmp_path, "train", 100)
    loader = DataLoader(tmp_path, "train", B=2, T=4)

    first = loader.next_batch()
    loader.reset()
    after_reset = loader.next_batch()

    assert torch.equal(after_reset[0], first[0])
    assert torch.equal(after_reset[1], first[1])


def test_too_small_file_is_rejected(tmp_path: Path) -> None:
    write_tokens(tmp_path, "train", 10)

    with pytest.raises(ValueError, match=r"10.*33"):
        DataLoader(tmp_path, "train", B=4, T=8)


def test_unknown_split_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DataLoader(tmp_path, "test", B=2, T=4)


def test_prepared_batch_is_consumable_by_gpt(tmp_path: Path) -> None:
    data_dir = tmp_path / "prepared"
    prepare(SAMPLE_TEXT, data_dir)
    loader = DataLoader(data_dir, "train", B=2, T=4)
    model = GPT(
        GPTConfig(
            block_size=16,
            vocab_size=50257,
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
    )

    x, y = loader.next_batch()
    _, loss = model(x, y)

    assert loss is not None
    assert torch.isfinite(loss)


@pytest.mark.network
def test_download_tinyshakespeare(tmp_path: Path) -> None:
    path = download_text(TINYSHAKESPEARE_URL, tmp_path / "input.txt")

    assert path.stat().st_size > 1_000_000
    assert "First Citizen" in path.read_text(encoding="utf-8")
