import datetime
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from gpt import GPT, GPTConfig
from gpt.data import DataLoader
from gpt.train import TrainConfig, configure_optimizers, evaluate, get_lr, train


def tiny_model_config() -> GPTConfig:
    return GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )


def write_synthetic_data(data_dir: Path, count: int = 4096) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens = (np.arange(count) % 100).astype(np.uint16)
    tokens.tofile(data_dir / "train.bin")
    tokens.tofile(data_dir / "val.bin")


def read_records(log_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def training_config(
    data_dir: Path,
    out_dir: Path,
    max_steps: int,
    eval_interval: int,
    seed: int = 1337,
) -> TrainConfig:
    return TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        model_config=tiny_model_config(),
        micro_batch_size=2,
        sequence_length=8,
        total_batch_size=16,
        max_steps=max_steps,
        warmup_steps=5,
        max_lr=3e-3,
        min_lr_ratio=0.1,
        weight_decay=0.0,
        eval_interval=eval_interval,
        eval_steps=2,
        device="cpu",
        use_amp=False,
        seed=seed,
    )


@pytest.fixture(scope="module")
def sixty_step_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TrainConfig, Path, list[dict[str, object]]]:
    root = tmp_path_factory.mktemp("sixty_step_run")
    data_dir = root / "data"
    write_synthetic_data(data_dir)
    config = training_config(data_dir, root / "out", max_steps=60, eval_interval=100)
    log_path = train(config)
    return config, log_path, read_records(log_path)


@pytest.fixture(scope="module")
def validation_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TrainConfig, list[dict[str, object]]]:
    root = tmp_path_factory.mktemp("validation_run")
    data_dir = root / "data"
    write_synthetic_data(data_dir)
    config = training_config(data_dir, root / "out", max_steps=20, eval_interval=10)
    records = read_records(train(config))
    return config, records


def test_get_lr_schedule_shape() -> None:
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 100

    assert get_lr(0, warmup_steps, max_steps, max_lr, min_lr) == pytest.approx(6e-5)
    assert get_lr(9, warmup_steps, max_steps, max_lr, min_lr) == pytest.approx(6e-4)
    assert get_lr(10, warmup_steps, max_steps, max_lr, min_lr) == pytest.approx(6e-4)
    assert get_lr(50, warmup_steps, max_steps, max_lr, min_lr) == pytest.approx(
        3.77e-4, rel=1e-2
    )
    assert min_lr < get_lr(99, warmup_steps, max_steps, max_lr, min_lr) < min_lr * 1.01
    assert get_lr(100, warmup_steps, max_steps, max_lr, min_lr) == min_lr
    assert get_lr(150, warmup_steps, max_steps, max_lr, min_lr) == min_lr

    warmup_lrs = [
        get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for step in range(warmup_steps)
    ]
    decay_lrs = [
        get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for step in range(warmup_steps, 150)
    ]
    assert all(left <= right for left, right in zip(warmup_lrs, warmup_lrs[1:]))
    assert all(left >= right for left, right in zip(decay_lrs, decay_lrs[1:]))
    assert all(
        min_lr <= get_lr(step, warmup_steps, max_steps, max_lr, min_lr) <= max_lr
        for step in range(150)
    )


def test_configure_optimizers_groups_parameters() -> None:
    model = GPT(tiny_model_config())
    optimizer = configure_optimizers(
        model,
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    assert len(optimizer.param_groups) == 2
    decay_group = next(
        group for group in optimizer.param_groups if group["weight_decay"] == 0.1
    )
    no_decay_group = next(
        group for group in optimizer.param_groups if group["weight_decay"] == 0.0
    )
    decay_ids = {id(parameter) for parameter in decay_group["params"]}
    no_decay_ids = {id(parameter) for parameter in no_decay_group["params"]}
    expected_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == {id(parameter) for parameter in expected_parameters}
    assert sum(parameter.numel() for parameter in decay_group["params"]) + sum(
        parameter.numel() for parameter in no_decay_group["params"]
    ) == sum(parameter.numel() for parameter in expected_parameters)
    assert all(parameter.dim() >= 2 for parameter in decay_group["params"])
    assert all(parameter.dim() < 2 for parameter in no_decay_group["params"])
    assert optimizer.defaults.get("fused") in (None, False)


def test_gradient_accumulation_matches_large_batch() -> None:
    torch.manual_seed(0)
    config = tiny_model_config()
    large_batch_model = GPT(config)
    accumulated_model = GPT(config)
    accumulated_model.load_state_dict(large_batch_model.state_dict())
    micro_batches = 4
    micro_batch_size = 2
    idx = torch.randint(0, config.vocab_size, (micro_batches * micro_batch_size, 8))
    targets = torch.randint(
        0, config.vocab_size, (micro_batches * micro_batch_size, 8)
    )

    _, large_loss = large_batch_model(idx, targets)
    assert large_loss is not None
    large_loss.backward()

    for micro_step in range(micro_batches):
        start = micro_step * micro_batch_size
        end = start + micro_batch_size
        _, micro_loss = accumulated_model(idx[start:end], targets[start:end])
        assert micro_loss is not None
        (micro_loss / micro_batches).backward()

    for large_parameter, accumulated_parameter in zip(
        large_batch_model.parameters(), accumulated_model.parameters()
    ):
        assert large_parameter.grad is not None
        assert accumulated_parameter.grad is not None
        difference = (large_parameter.grad - accumulated_parameter.grad).abs().max()
        assert difference.item() < 1e-6


def test_training_reduces_loss(
    sixty_step_run: tuple[TrainConfig, Path, list[dict[str, object]]],
) -> None:
    _, _, records = sixty_step_run
    losses = [float(record["loss"]) for record in records if record["type"] == "train"]

    first_mean = sum(losses[:5]) / 5
    last_mean = sum(losses[-5:]) / 5

    assert last_mean < first_mean - 0.5


def test_log_file_contains_one_record_per_step(
    sixty_step_run: tuple[TrainConfig, Path, list[dict[str, object]]],
) -> None:
    config, log_path, records = sixty_step_run
    assert log_path.exists()
    train_records = [record for record in records if record["type"] == "train"]
    required_fields = {
        "step",
        "loss",
        "lr",
        "grad_norm",
        "dt_ms",
        "tokens_per_sec",
        "tokens_seen",
    }

    assert len(train_records) == config.max_steps
    assert [record["step"] for record in train_records] == list(
        range(config.max_steps)
    )
    assert all(required_fields <= record.keys() for record in train_records)


def test_tokens_seen_accounting(
    sixty_step_run: tuple[TrainConfig, Path, list[dict[str, object]]],
) -> None:
    config, _, records = sixty_step_run
    final_record = [record for record in records if record["type"] == "train"][-1]

    assert final_record["tokens_seen"] == (
        int(final_record["step"]) + 1
    ) * config.total_batch_size


def test_validation_records_and_training_mode(
    validation_run: tuple[TrainConfig, list[dict[str, object]]], tmp_path: Path
) -> None:
    config, records = validation_run
    val_records = [record for record in records if record["type"] == "val"]

    assert [record["step"] for record in val_records] == [0, 10, 20]
    assert val_records[-1]["step"] == config.max_steps
    assert all(math.isfinite(float(record["loss"])) for record in val_records)

    data_dir = tmp_path / "data"
    write_synthetic_data(data_dir)
    model = GPT(tiny_model_config())
    loader = DataLoader(data_dir, "val", B=2, T=8)
    evaluate(model, loader, steps=2, device="cpu", use_amp=False)
    assert model.training is True


def test_evaluate_is_repeatable(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_synthetic_data(data_dir)
    torch.manual_seed(1)
    model = GPT(tiny_model_config())
    loader = DataLoader(data_dir, "val", B=2, T=8)

    first = evaluate(model, loader, steps=3, device="cpu", use_amp=False)
    second = evaluate(model, loader, steps=3, device="cpu", use_amp=False)

    assert first == second


def test_train_config_rejects_indivisible_batch_size() -> None:
    with pytest.raises(ValueError, match=r"25.*3.*8"):
        TrainConfig(
            micro_batch_size=3,
            sequence_length=8,
            total_batch_size=25,
        )


def test_gradient_clipping_limits_post_clip_norm() -> None:
    torch.manual_seed(2)
    model = GPT(tiny_model_config())
    idx = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 100, (2, 8))
    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    pre_clip_norm = torch.sqrt(
        sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)
    )
    clip = 1e-3

    returned_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    post_clip_norm = torch.sqrt(
        sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)
    )

    assert post_clip_norm.item() <= clip + 1e-6
    assert returned_norm.item() == pytest.approx(pre_clip_norm.item())


def test_training_is_deterministic(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_synthetic_data(data_dir)
    first_config = training_config(
        data_dir, tmp_path / "out1", max_steps=5, eval_interval=10, seed=42
    )
    second_config = training_config(
        data_dir, tmp_path / "out2", max_steps=5, eval_interval=10, seed=42
    )

    first_records = read_records(train(first_config))
    second_records = read_records(train(second_config))
    first_losses = [
        record["loss"] for record in first_records if record["type"] == "train"
    ]
    second_losses = [
        record["loss"] for record in second_records if record["type"] == "train"
    ]

    assert first_losses == second_losses


def test_run_marker_delimits_runs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_synthetic_data(data_dir)
    config = training_config(
        data_dir, out_dir, max_steps=3, eval_interval=10, seed=42
    )

    train(config)
    records = read_records(train(config))
    run_markers = [record for record in records if record["type"] == "run"]

    assert len(run_markers) == 2
    assert records[0]["type"] == "run"

    groups: list[list[dict[str, object]]] = []
    for record in records:
        if record["type"] == "run":
            groups.append([])
        else:
            groups[-1].append(record)

    assert len(groups) == 2
    for group in groups:
        train_records = [record for record in group if record["type"] == "train"]
        assert len(train_records) == config.max_steps
        assert [record["step"] for record in train_records] == list(
            range(config.max_steps)
        )

    for marker in run_markers:
        assert marker["max_steps"] == config.max_steps
        assert marker["total_batch_size"] == config.total_batch_size
        assert marker["seed"] == config.seed
        assert marker["n_layer"] == config.model_config.n_layer
        assert marker["vocab_size"] == config.model_config.vocab_size
        datetime.datetime.fromisoformat(str(marker["started"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_training_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_synthetic_data(data_dir)
    config = training_config(
        data_dir, tmp_path / "out", max_steps=2, eval_interval=10
    )
    config.device = "cuda"
    config.use_amp = True

    records = read_records(train(config))
    losses = [float(record["loss"]) for record in records if record["type"] == "train"]

    assert len(losses) == 2
    assert all(math.isfinite(loss) for loss in losses)
