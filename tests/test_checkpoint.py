from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gpt import GPT, GPTConfig
from gpt.checkpoint import load_checkpoint, save_checkpoint
from gpt.train import TrainConfig, configure_optimizers, get_lr, train


def tiny_model_config(n_layer: int = 2) -> GPTConfig:
    return GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=n_layer,
        n_head=4,
        n_embd=64,
    )


def write_synthetic_data(data_dir: Path, count: int = 4096) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens = (np.arange(count) % 100).astype(np.uint16)
    tokens.tofile(data_dir / "train.bin")
    tokens.tofile(data_dir / "val.bin")


def make_optimizer(model: GPT, device_type: str = "cpu") -> torch.optim.AdamW:
    return configure_optimizers(
        model,
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type=device_type,
    )


def populate_optimizer_state(model: GPT, optimizer: torch.optim.Optimizer) -> None:
    device = next(model.parameters()).device
    idx = torch.randint(0, model.config.vocab_size, (2, 8), device=device)
    targets = torch.randint(
        0,
        model.config.vocab_size,
        (2, 8),
        device=device,
    )
    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def training_config(
    data_dir: Path,
    out_dir: Path,
    max_steps: int,
    *,
    warmup_steps: int = 1,
    min_lr_ratio: float = 1.0,
    checkpoint_interval: int = 0,
    snapshot_interval: int = 0,
    resume: bool = False,
    model_config: GPTConfig | None = None,
) -> TrainConfig:
    return TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        model_config=model_config or tiny_model_config(),
        micro_batch_size=2,
        sequence_length=8,
        total_batch_size=16,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        max_lr=6e-4,
        min_lr_ratio=min_lr_ratio,
        weight_decay=0.1,
        eval_interval=100,
        eval_steps=2,
        device="cpu",
        use_amp=False,
        seed=1337,
        checkpoint_interval=checkpoint_interval,
        snapshot_interval=snapshot_interval,
        resume=resume,
    )


def read_records(log_path: Path) -> list[dict[str, object]]:
    import json

    return [json.loads(line) for line in log_path.read_text().splitlines()]


def assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = tiny_model_config()
    model = GPT(config)
    optimizer = make_optimizer(model)
    populate_optimizer_state(model, optimizer)
    path = tmp_path / "nested" / "ckpt.pt"
    train_config = {"data_dir": tmp_path / "data", "max_steps": 6}
    model_config = asdict(config)

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=3,
        train_position=48,
        val_loss=4.25,
        train_config=train_config,
        model_config=model_config,
    )
    checkpoint = load_checkpoint(path)

    assert set(checkpoint) == {
        "model",
        "optimizer",
        "step",
        "train_position",
        "val_loss",
        "train_config",
        "model_config",
    }
    assert checkpoint["step"] == 3
    assert checkpoint["train_position"] == 48
    assert checkpoint["val_loss"] == 4.25
    assert checkpoint["train_config"] == {
        "data_dir": str(tmp_path / "data"),
        "max_steps": 6,
    }
    assert checkpoint["model_config"] == model_config

    restored_model = GPT(config)
    restored_optimizer = make_optimizer(restored_model)
    restored_model.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    assert_nested_equal(model.state_dict(), restored_model.state_dict())
    assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())


def test_checkpoint_loads_with_safe_default(tmp_path: Path) -> None:
    config = tiny_model_config()
    model = GPT(config)
    optimizer = make_optimizer(model)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=0,
        train_position=0,
        val_loss=float("nan"),
        train_config={"data_dir": tmp_path / "data", "resume": False},
        model_config=asdict(config),
    )

    checkpoint = torch.load(path)
    saved_train_config = checkpoint["train_config"]
    allowed_types = (int, float, str, bool, list, dict, type(None))

    assert all(
        isinstance(value, allowed_types) for value in saved_train_config.values()
    )
    assert isinstance(saved_train_config["data_dir"], str)


def test_resume_is_bit_exact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_synthetic_data(data_dir)

    straight_config = training_config(data_dir, tmp_path / "out_a", max_steps=6)
    straight_records = read_records(train(straight_config))

    split_out_dir = tmp_path / "out_b"
    first_phase = training_config(data_dir, split_out_dir, max_steps=3)
    train(first_phase)
    resumed_config = replace(first_phase, max_steps=6, resume=True)
    resumed_records = read_records(train(resumed_config))

    straight_losses = [
        record["loss"] for record in straight_records if record["type"] == "train"
    ]
    resumed_losses = [
        record["loss"] for record in resumed_records if record["type"] == "train"
    ]
    assert len(straight_losses) == len(resumed_losses) == 6
    assert straight_losses == resumed_losses


def test_resume_continues_steps_and_schedule(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_synthetic_data(data_dir)
    first_phase = training_config(
        data_dir,
        out_dir,
        max_steps=3,
        warmup_steps=2,
        min_lr_ratio=0.1,
    )
    train(first_phase)
    resumed_config = replace(first_phase, max_steps=6, resume=True)
    records = read_records(train(resumed_config))

    train_records = [record for record in records if record["type"] == "train"]
    run_records = [record for record in records if record["type"] == "run"]

    assert [record["step"] for record in train_records] == [0, 1, 2, 3, 4, 5]
    assert run_records[1]["resumed_from"] == 3
    for record in train_records[3:]:
        step = int(record["step"])
        assert record["lr"] == get_lr(
            step,
            resumed_config.warmup_steps,
            resumed_config.max_steps,
            resumed_config.max_lr,
            resumed_config.max_lr * resumed_config.min_lr_ratio,
        )


def test_checkpoint_write_is_atomic_and_replaces(tmp_path: Path) -> None:
    config = tiny_model_config()
    model = GPT(config)
    optimizer = make_optimizer(model)
    path = tmp_path / "ckpt.pt"
    arguments = {
        "model": model,
        "optimizer": optimizer,
        "train_position": 16,
        "train_config": {"max_steps": 2},
        "model_config": asdict(config),
    }

    save_checkpoint(path, step=1, val_loss=4.0, **arguments)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()

    save_checkpoint(path, step=2, val_loss=3.5, **arguments)
    checkpoint = load_checkpoint(path)
    assert checkpoint["step"] == 2
    assert checkpoint["val_loss"] == 3.5
    assert not path.with_suffix(".tmp").exists()


def test_resume_rejects_model_config_mismatch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_synthetic_data(data_dir)
    train(training_config(data_dir, out_dir, max_steps=1))
    resume_config = training_config(
        data_dir,
        out_dir,
        max_steps=2,
        resume=True,
        model_config=tiny_model_config(n_layer=4),
    )

    with pytest.raises(
        ValueError,
        match=r"model_config mismatch.*n_layer.*saved 2.*current 4",
    ):
        train(resume_config)


def test_resume_rejects_batch_shape_mismatch(tmp_path: Path) -> None:
    mismatches = {
        "micro_batch_size": 1,
        "sequence_length": 4,
        "total_batch_size": 32,
    }
    for field_name, new_value in mismatches.items():
        case_dir = tmp_path / field_name
        data_dir = case_dir / "data"
        out_dir = case_dir / "out"
        write_synthetic_data(data_dir)
        first_config = training_config(data_dir, out_dir, max_steps=1)
        train(first_config)
        resume_config = replace(
            first_config,
            max_steps=2,
            resume=True,
            **{field_name: new_value},
        )

        with pytest.raises(ValueError, match=rf"{field_name} mismatch"):
            train(resume_config)


def test_resume_without_checkpoint_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "missing"
    write_synthetic_data(data_dir)
    config = training_config(data_dir, out_dir, max_steps=2, resume=True)

    with pytest.raises(FileNotFoundError) as error:
        train(config)

    assert str(out_dir / "ckpt.pt") in str(error.value)
    assert "without --resume" in str(error.value)


def test_final_checkpoint_is_always_written(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_synthetic_data(data_dir)
    config = training_config(
        data_dir,
        out_dir,
        max_steps=3,
        checkpoint_interval=1000,
    )

    train(config)
    checkpoint = load_checkpoint(out_dir / "ckpt.pt")

    assert checkpoint["step"] == 3


def test_checkpoint_interval_updates_rolling_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_synthetic_data(data_dir)
    config = training_config(
        data_dir,
        out_dir,
        max_steps=4,
        checkpoint_interval=2,
    )

    train(config)
    checkpoint = load_checkpoint(out_dir / "ckpt.pt")

    assert checkpoint["step"] == 4


def test_snapshot_interval_uses_completed_steps(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_synthetic_data(data_dir)
    config = training_config(
        data_dir,
        out_dir,
        max_steps=4,
        snapshot_interval=2,
    )

    train(config)

    checkpoint_names = sorted(path.name for path in out_dir.glob("ckpt*.pt"))
    assert checkpoint_names == [
        "ckpt.pt",
        "ckpt_000002.pt",
        "ckpt_000004.pt",
    ]
    assert load_checkpoint(out_dir / "ckpt_000002.pt")["step"] == 2
    assert load_checkpoint(out_dir / "ckpt_000004.pt")["step"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_checkpoint_loads_on_cpu(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = tiny_model_config()
    model = GPT(config).to("cuda")
    optimizer = make_optimizer(model, device_type="cuda")
    populate_optimizer_state(model, optimizer)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        train_position=16,
        val_loss=4.0,
        train_config={"device": "cuda"},
        model_config=asdict(config),
    )
    checkpoint = load_checkpoint(path, map_location="cpu")

    assert all(
        tensor.device.type == "cpu" for tensor in checkpoint["model"].values()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_resume_does_not_retain_checkpoint_on_gpu(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_synthetic_data(data_dir)
    first_config = training_config(data_dir, out_dir, max_steps=2)
    first_config.device = "cuda"
    first_config.use_amp = True
    train(first_config)

    resumed_config = replace(first_config, max_steps=4, resume=True)
    train(resumed_config)

    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() < 100 * 1024**2
