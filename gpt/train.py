"""The GPT-2 training loop with AdamW, warmup plus cosine learning-rate decay,
gradient accumulation, gradient clipping, JSONL logging, and periodic validation.

Includes periodic validation and JSONL metrics. For a short smoke test, run:

    python -m gpt.train --max-steps 20 --total-batch-size 8192 --eval-interval 10
"""

import argparse
import datetime
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .config import GPTConfig
from .data import DataLoader
from .model import GPT


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    out_dir: Path = Path("out")
    model_config: GPTConfig = field(default_factory=GPTConfig)

    micro_batch_size: int = 4
    sequence_length: int = 1024
    total_batch_size: int = 524_288

    max_steps: int = 19_073
    warmup_steps: int = 715
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_interval: int = 250
    eval_steps: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    compile_model: bool = False
    seed: int = 1337

    checkpoint_interval: int = 1000
    snapshot_interval: int = 0
    resume: bool = False

    def __post_init__(self) -> None:
        micro_batch_tokens = self.micro_batch_size * self.sequence_length
        if self.total_batch_size % micro_batch_tokens != 0:
            raise ValueError(
                f"total_batch_size {self.total_batch_size} must be divisible by "
                f"micro_batch_size {self.micro_batch_size} * "
                f"sequence_length {self.sequence_length}"
            )


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (
        max_lr - min_lr
    )


def configure_optimizers(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.AdamW:
    parameters = [
        parameter for _, parameter in model.named_parameters() if parameter.requires_grad
    ]
    decay_parameters = [parameter for parameter in parameters if parameter.dim() >= 2]
    no_decay_parameters = [parameter for parameter in parameters if parameter.dim() < 2]
    parameter_groups = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    optimizer_kwargs: dict[str, object] = {
        "lr": learning_rate,
        "betas": betas,
        "eps": 1e-8,
    }
    if device_type == "cuda":
        optimizer_kwargs["fused"] = True
    return torch.optim.AdamW(parameter_groups, **optimizer_kwargs)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    steps: int,
    device: str,
    use_amp: bool,
) -> float:
    device_type = torch.device(device).type
    amp = torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=use_amp and device_type == "cuda",
    )
    model.eval()
    loader.reset()
    loss_accum = torch.zeros((), device=device)
    with torch.no_grad():
        for _ in range(steps):
            x, y = loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp:
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("model did not return a validation loss")
            loss_accum += loss.detach()
    model.train()
    return (loss_accum / steps).item()


def train(config: TrainConfig) -> Path:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    device_type = torch.device(config.device).type

    config.out_dir.mkdir(parents=True, exist_ok=True)
    grad_accum_steps = config.total_batch_size // (
        config.micro_batch_size * config.sequence_length
    )
    train_loader = DataLoader(
        config.data_dir,
        "train",
        B=config.micro_batch_size,
        T=config.sequence_length,
    )
    val_loader = DataLoader(
        config.data_dir,
        "val",
        B=config.micro_batch_size,
        T=config.sequence_length,
    )
    model: nn.Module = GPT(config.model_config).to(config.device)
    optimizer = configure_optimizers(
        model,
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )
    min_lr = config.max_lr * config.min_lr_ratio
    rolling_checkpoint_path = config.out_dir / "ckpt.pt"
    train_config_dict = asdict(config)
    model_config_dict = asdict(config.model_config)
    start_step = 0
    last_val_loss = float("nan")

    if config.resume:
        try:
            checkpoint = load_checkpoint(
                rolling_checkpoint_path,
                map_location="cpu",
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"resume checkpoint not found at {rolling_checkpoint_path}; "
                "start the run without --resume"
            ) from exc

        saved_model_config = checkpoint["model_config"]
        if not isinstance(saved_model_config, dict):
            raise ValueError("saved model_config is not a dictionary")
        if saved_model_config != model_config_dict:
            differing_fields = sorted(
                set(saved_model_config) | set(model_config_dict)
            )
            differences = [
                f"{field_name}: saved {saved_model_config.get(field_name)!r}, "
                f"current {model_config_dict.get(field_name)!r}"
                for field_name in differing_fields
                if saved_model_config.get(field_name)
                != model_config_dict.get(field_name)
            ]
            raise ValueError("model_config mismatch: " + "; ".join(differences))

        saved_train_config = checkpoint["train_config"]
        if not isinstance(saved_train_config, dict):
            raise ValueError("saved train_config is not a dictionary")
        for field_name in (
            "micro_batch_size",
            "sequence_length",
            "total_batch_size",
        ):
            saved_value = saved_train_config.get(field_name)
            current_value = getattr(config, field_name)
            if saved_value != current_value:
                raise ValueError(
                    f"{field_name} mismatch: saved {saved_value!r}, "
                    f"current {current_value!r}"
                )

        for field_name in (
            "max_steps",
            "warmup_steps",
            "max_lr",
            "min_lr_ratio",
        ):
            saved_value = saved_train_config.get(field_name)
            current_value = getattr(config, field_name)
            if saved_value != current_value:
                print(
                    f"warning: {field_name} changed from saved {saved_value!r} "
                    f"to current {current_value!r}"
                )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        train_position = int(checkpoint["train_position"])
        required_tokens = config.micro_batch_size * config.sequence_length + 1
        if train_position + required_tokens > len(train_loader.tokens):
            print(
                f"checkpoint train_position {train_position} cannot provide the "
                f"next {required_tokens} tokens; resetting it to 0"
            )
            train_position = 0
        train_loader.position = train_position
        last_val_loss = float(checkpoint["val_loss"])
        del checkpoint

    if config.compile_model:
        model = torch.compile(model)

    amp = torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=config.use_amp and device_type == "cuda",
    )
    log_path = config.out_dir / "log.jsonl"

    with log_path.open("a", encoding="utf-8") as log_file:
        run_record = {
            "type": "run",
            "started": datetime.datetime.now(datetime.UTC).isoformat(),
            "max_steps": config.max_steps,
            "total_batch_size": config.total_batch_size,
            "micro_batch_size": config.micro_batch_size,
            "sequence_length": config.sequence_length,
            "grad_accum_steps": grad_accum_steps,
            "max_lr": config.max_lr,
            "min_lr": min_lr,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "seed": config.seed,
            "device": config.device,
            "use_amp": config.use_amp,
            "n_layer": config.model_config.n_layer,
            "n_head": config.model_config.n_head,
            "n_embd": config.model_config.n_embd,
            "vocab_size": config.model_config.vocab_size,
            "block_size": config.model_config.block_size,
            "resumed_from": start_step if config.resume else None,
        }
        log_file.write(json.dumps(run_record) + "\n")
        log_file.flush()

        for step in range(start_step, config.max_steps):
            if step % config.eval_interval == 0:
                last_val_loss = evaluate(
                    model,
                    val_loader,
                    steps=config.eval_steps,
                    device=config.device,
                    use_amp=config.use_amp,
                )
                val_record = {
                    "type": "val",
                    "step": step,
                    "loss": last_val_loss,
                }
                log_file.write(json.dumps(val_record) + "\n")
                log_file.flush()

            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=config.device)
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x = x.to(config.device, non_blocking=True)
                y = y.to(config.device, non_blocking=True)
                with amp:
                    _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("model did not return a training loss")
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            lr = get_lr(
                step,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                max_lr=config.max_lr,
                min_lr=min_lr,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = lr
            optimizer.step()
            if device_type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_start

            loss_value = loss_accum.item()
            grad_norm_value = grad_norm.item()
            tokens_per_sec = config.total_batch_size / elapsed
            train_record = {
                "type": "train",
                "step": step,
                "loss": loss_value,
                "lr": lr,
                "grad_norm": grad_norm_value,
                "dt_ms": elapsed * 1000.0,
                "tokens_per_sec": tokens_per_sec,
                "tokens_seen": (step + 1) * config.total_batch_size,
            }
            log_file.write(json.dumps(train_record) + "\n")
            log_file.flush()
            print(
                f"step {step:5d} | loss {loss_value:.6f} | lr {lr:.4e} | "
                f"grad_norm {grad_norm_value:.4f} | "
                f"dt {elapsed * 1000.0:.2f} ms | tok/s {tokens_per_sec:.0f}"
            )

            completed_step = step + 1
            if (
                config.checkpoint_interval > 0
                and completed_step % config.checkpoint_interval == 0
            ) or step == config.max_steps - 1:
                save_checkpoint(
                    rolling_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=completed_step,
                    train_position=train_loader.position,
                    val_loss=last_val_loss,
                    train_config=train_config_dict,
                    model_config=model_config_dict,
                )
            if (
                config.snapshot_interval > 0
                and completed_step % config.snapshot_interval == 0
            ):
                snapshot_path = (
                    config.out_dir / f"ckpt_{completed_step:06d}.pt"
                )
                save_checkpoint(
                    snapshot_path,
                    model=model,
                    optimizer=optimizer,
                    step=completed_step,
                    train_position=train_loader.position,
                    val_loss=last_val_loss,
                    train_config=train_config_dict,
                    model_config=model_config_dict,
                )

        last_val_loss = evaluate(
            model,
            val_loader,
            steps=config.eval_steps,
            device=config.device,
            use_amp=config.use_amp,
        )
        val_record = {
            "type": "val",
            "step": config.max_steps,
            "loss": last_val_loss,
        }
        log_file.write(json.dumps(val_record) + "\n")
        log_file.flush()

    return log_path


def main(argv: list[str] | None = None) -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train GPT-2 from scratch")
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument(
        "--micro-batch-size", type=int, default=defaults.micro_batch_size
    )
    parser.add_argument("--sequence-length", type=int, default=defaults.sequence_length)
    parser.add_argument(
        "--total-batch-size", type=int, default=defaults.total_batch_size
    )
    parser.add_argument("--max-lr", type=float, default=defaults.max_lr)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval-steps", type=int, default=defaults.eval_steps)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=defaults.checkpoint_interval
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=defaults.snapshot_interval,
        help="retain a separate full-size checkpoint at this interval; increases disk use",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    args = parser.parse_args(argv)

    config = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        total_batch_size=args.total_batch_size,
        max_lr=args.max_lr,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        device=args.device,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        snapshot_interval=args.snapshot_interval,
        resume=args.resume,
        use_amp=not args.no_amp,
        compile_model=args.compile_model,
    )
    train(config)


if __name__ == "__main__":
    main()
