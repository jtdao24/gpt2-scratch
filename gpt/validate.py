"""Compare our pretrained GPT-2 with Hugging Face's GPT-2 as the reference.

    python -m gpt.validate
    python -m gpt.validate --device cuda --output results/validation.json
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import get_tokenizer
from .generate import generate
from .hf_weights import HF_MODEL_IDS, convert_state_dict, from_pretrained
from .model import MLP


LOGIT_SHAPES = ((1, 1), (1, 16), (2, 64), (4, 256), (2, 1024))
PROMPTS = (
    "The meaning of life is",
    "Hello, I am a language model,",
    "In 1492, Columbus",
)


@dataclass
class CheckResult:
    name: str
    status: str
    value: float | None
    threshold: str
    detail: str


def compare_logits(
    ours: torch.Tensor, ref: torch.Tensor
) -> dict[str, float]:
    """Measure float32 error and the fraction of positions with matching argmax."""
    ours, ref = ours.float(), ref.float()
    maximum = (ours - ref).abs_().amax().item()
    scale = ref.abs().amax().item()
    relative = maximum / scale if scale else (0.0 if maximum == 0.0 else math.inf)
    agreement = (ours.argmax(dim=-1) == ref.argmax(dim=-1)).float().mean().item()
    return {
        "max_abs_diff": maximum,
        "relative_diff": relative,
        "argmax_agreement": agreement,
    }


def aligned_loss(model: nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Score aligned next tokens in our GPT. HF shifts labels=idx internally."""
    return model(idx[:, :-1], targets=idx[:, 1:])[1]


def exit_code(results: list[CheckResult]) -> int:
    """Return a nonzero exit code on any failed check."""
    return int(any(result.status == "fail" for result in results))


def load_reference(
    device: str = "cpu", attn_implementation: str = "sdpa"
) -> nn.Module:
    """Load the reference model in evaluation mode."""
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(
        HF_MODEL_IDS["gpt2"], attn_implementation=attn_implementation
    ).to(device).eval()


@contextmanager
def _full_precision_matmul() -> Iterator[None]:
    """Keep FP32 validation independent of the training precision setting."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@torch.no_grad()
def check_architecture(ours: nn.Module, ref: nn.Module) -> CheckResult:
    mismatches: list[str] = []
    for our_field, ref_field in (
        ("n_layer", "n_layer"),
        ("n_head", "n_head"),
        ("n_embd", "n_embd"),
        ("block_size", "n_positions"),
        ("vocab_size", "vocab_size"),
    ):
        actual = getattr(ours.config, our_field)
        expected = getattr(ref.config, ref_field)
        if actual != expected:
            mismatches.append(f"{our_field}={actual} vs reference {ref_field}={expected}")
    for name, module in ours.named_modules():
        if isinstance(module, nn.LayerNorm):
            expected = ref.config.layer_norm_epsilon
            if module.eps != expected:
                mismatches.append(f"{name}.eps={module.eps} vs reference {expected}")
        if isinstance(module, MLP) and (
            not isinstance(module.gelu, nn.GELU) or module.gelu.approximate != "tanh"
        ):
            mismatches.append(f"{name}.gelu must use the tanh approximation")
    if ref.config.activation_function != "gelu_new":
        mismatches.append(f"reference activation_function={ref.config.activation_function}")
    if ref.config.tie_word_embeddings is not True:
        mismatches.append("reference tie_word_embeddings must be True")
    if ours.lm_head.weight is not ours.transformer.wte.weight:
        mismatches.append("our lm_head.weight and transformer.wte.weight must be tied")
    return CheckResult(
        "architecture", "fail" if mismatches else "pass", None, "exact match",
        "; ".join(mismatches) or "Model dimensions, LayerNorm, GELU, and weight tying agree",
    )


@torch.no_grad()
def check_parameters(ours: nn.Module, ref: nn.Module) -> CheckResult:
    count = sum(parameter.numel() for parameter in ours.parameters())
    our_state = ours.state_dict()
    ref_state = convert_state_dict(ref.state_dict())
    missing = ref_state.keys() - our_state.keys()
    extra = our_state.keys() - ref_state.keys()
    different = [
        key for key in sorted(our_state.keys() & ref_state.keys())
        if not torch.equal(our_state[key], ref_state[key])
    ]
    mismatch_count = len(missing) + len(extra) + len(different)
    details = [f"{count:,} parameters and {len(our_state)} tensors"]
    if count != 124_439_808:
        details.append("expected 124,439,808 parameters")
    for label, keys in (
        ("missing", sorted(missing)),
        ("extra", sorted(extra)),
        ("different", different),
    ):
        if keys:
            details.append(f"{label} keys: {', '.join(keys)}")
    passed = mismatch_count == 0 and count == 124_439_808
    return CheckResult(
        "parameters", "pass" if passed else "fail", float(mismatch_count),
        "0 mismatches", "; ".join(details),
    )


def _random_ids(
    shape: tuple[int, int], device: str, generator: torch.Generator | None = None
) -> torch.Tensor:
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)
    return torch.randint(
        50257, shape, device=device, dtype=torch.int64, generator=generator
    )


def _logit_result(name: str, ours: torch.Tensor, ref: torch.Tensor) -> CheckResult:
    metrics = compare_logits(ours, ref)
    absolute = metrics["max_abs_diff"]
    relative = metrics["relative_diff"]
    agreement = metrics["argmax_agreement"]
    passed = absolute < 1e-3 and relative < 1e-5 and agreement == 1.0
    return CheckResult(
        name, "pass" if passed else "fail", absolute,
        "abs < 1e-3, rel < 1e-5, argmax = 100%",
        f"relative={relative:.3g}, argmax={agreement:.6%}",
    )


@torch.no_grad()
def check_logits(ours: nn.Module, ref: nn.Module, device: str) -> list[CheckResult]:
    generator = torch.Generator(device=device).manual_seed(0)
    results: list[CheckResult] = []
    for batch, length in LOGIT_SHAPES:
        ids = _random_ids((batch, length), device, generator)
        results.append(
            _logit_result(f"logits ({batch}, {length})", ours(ids)[0], ref(ids).logits)
        )
    return results


@torch.no_grad()
def check_loss(ours: nn.Module, ref: nn.Module, device: str) -> CheckResult:
    ids = _random_ids((2, 1024), device)
    our_loss = aligned_loss(ours, ids).item()
    ref_loss = ref(ids, labels=ids).loss.item()
    difference = abs(our_loss - ref_loss)
    return CheckResult(
        "aligned loss", "pass" if difference < 1e-5 else "fail", difference, "< 1e-5",
        f"ours={our_loss:.8f}, reference={ref_loss:.8f}, 1023 pairs per batch row",
    )


@torch.no_grad()
def check_text_loss(
    ours: nn.Module, ref: nn.Module, data_dir: Path, device: str
) -> CheckResult:
    path = data_dir / "val.bin"
    if not path.is_file():
        return CheckResult("text loss", "skip", None, "abs diff < 1e-5", f"{path} missing")

    tokens = np.fromfile(path, dtype=np.uint16)
    window_size = 1025
    count = len(tokens) // window_size
    if count == 0:
        return CheckResult(
            "text loss", "skip", None, "abs diff < 1e-5", f"{path} has no complete windows"
        )
    windows = torch.from_numpy(
        tokens[:count * window_size].astype(np.int64).reshape(count, window_size)
    ).to(device)
    our_losses: list[float] = []
    ref_losses: list[float] = []
    for window in windows.split(1):
        our_losses.append(aligned_loss(ours, window).item())
        logits = ref(window[:, :-1]).logits
        ref_losses.append(
            F.cross_entropy(logits.flatten(0, 1), window[:, 1:].reshape(-1)).item()
        )
    our_mean = math.fsum(our_losses) / count
    ref_mean = math.fsum(ref_losses) / count
    difference = abs(our_mean - ref_mean)
    return CheckResult(
        "text loss", "pass" if difference < 1e-5 else "fail", our_mean, "abs diff < 1e-5",
        f"{count} windows, reference={ref_mean:.6f}, diff={difference:.3g}, "
        f"perplexity={math.exp(our_mean):.2f}",
    )


@torch.no_grad()
def check_generation(ours: nn.Module, ref: nn.Module, device: str) -> list[CheckResult]:
    tokenizer = get_tokenizer()
    results: list[CheckResult] = []
    for prompt in PROMPTS:
        ids = torch.tensor(
            [tokenizer.encode_ordinary(prompt)], dtype=torch.int64, device=device
        )
        actual = generate(ours, ids, 50, top_k=1)[0].tolist()
        expected = ref.generate(
            ids, max_new_tokens=50, do_sample=False, pad_token_id=50256
        )[0].tolist()
        passed = actual == expected
        if passed:
            detail = f"{len(expected) - ids.size(1)} generated tokens match"
        else:
            position = next(
                (index for index, (a, b) in enumerate(zip(actual, expected)) if a != b),
                min(len(actual), len(expected)),
            )
            prefix = torch.tensor([expected[:position]], dtype=torch.int64, device=device)
            top_two = ref(prefix).logits[0, -1].topk(2).values
            margin = (top_two[0] - top_two[1]).item()
            detail = (
                f"first divergence at position {position} (0-based), "
                f"HF top-1/top-2 margin={margin:.6g}"
            )
        results.append(CheckResult(
            f"generation: {prompt}", "pass" if passed else "fail", None, "exact tokens", detail
        ))
    return results


@torch.no_grad()
@_full_precision_matmul()
def check_gpu(ours: nn.Module, ref: nn.Module) -> list[CheckResult]:
    if not torch.cuda.is_available():
        return [CheckResult("GPU", "skip", None, "", "CUDA unavailable")]

    our_device = next(ours.parameters()).device
    ref_device = next(ref.parameters()).device
    device = str(our_device) if our_device.type == "cuda" else "cuda"
    try:
        ours.to(device).eval()
        ref.to(device).eval()
        ids = _random_ids((2, 1024), device)
        reference_fp32 = ref(ids, labels=ids)
        results = [_logit_result("GPU fp32", ours(ids)[0], reference_fp32.logits)]
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                results.append(CheckResult("GPU bf16", "skip", None, "", "CUDA bf16 unavailable"))
                return results
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            our_bf16 = ours(ids)[0]
            reference_bf16 = ref(ids, labels=ids)
            our_loss = aligned_loss(ours, ids).item()

        same_precision = compare_logits(our_bf16, reference_bf16.logits)
        agreement = same_precision["argmax_agreement"]
        loss_difference = abs(our_loss - reference_bf16.loss.item())
        passed = agreement >= 0.99 and loss_difference < 1e-2
        results.append(CheckResult(
            "GPU bf16", "pass" if passed else "fail", agreement,
            "argmax >= 99%, loss diff < 1e-2",
            f"argmax={agreement:.6%}, loss diff={loss_difference:.3g}, "
            f"max abs diff={same_precision['max_abs_diff']:.3g}",
        ))
        precision_difference = compare_logits(our_bf16, reference_fp32.logits)["max_abs_diff"]
        results.append(CheckResult(
            "GPU bf16 vs fp32", "info", precision_difference, "",
            f"bf16 precision cost, loss diff={abs(our_loss - reference_fp32.loss.item()):.3g}",
        ))
        return results
    finally:
        ours.to(our_device)
        ref.to(ref_device)


@torch.no_grad()
def check_eager_reference(ours: nn.Module, device: str) -> CheckResult:
    ref = load_reference(device, attn_implementation="eager")
    ids = _random_ids((2, 256), device)
    return _logit_result("eager reference", ours(ids)[0], ref(ids).logits)


@_full_precision_matmul()
def run_validation(
    device: str = "cpu",
    data_dir: Path = Path("data"),
    ours: nn.Module | None = None,
    ref: nn.Module | None = None,
) -> list[CheckResult]:
    """Run all checks, using supplied models when provided."""
    if ours is None:
        ours = from_pretrained("gpt2", device)
    if ref is None:
        ref = load_reference(device)
    ours.to(device=device, dtype=torch.float32).eval()
    ref.to(device=device, dtype=torch.float32).eval()
    return [
        check_architecture(ours, ref),
        check_parameters(ours, ref),
        *check_logits(ours, ref, device),
        check_loss(ours, ref, device),
        check_text_loss(ours, ref, data_dir, device),
        *check_generation(ours, ref, device),
        *check_gpu(ours, ref),
        check_eager_reference(ours, device),
    ]


def write_report(results: list[CheckResult], path: Path, *, device: str = "cpu") -> None:
    """Write the checks and their environment as JSON."""
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        transformers_version = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    report_device = torch.device(device)
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "python": platform.python_version(),
        "device": (
            torch.cuda.get_device_name(report_device)
            if report_device.type == "cuda" else str(report_device)
        ),
        "git_commit": commit,
        "checks": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate GPT-2 against its Hugging Face reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    results = run_validation(device=args.device, data_dir=args.data_dir)
    print(f"{'status':<6} {'name':<42} {'value':>11}  {'threshold':<39} detail")
    for result in results:
        value = "-" if result.value is None else f"{result.value:.6g}"
        print(f"{result.status:<6} {result.name:<42} {value:>11}  "
              f"{result.threshold or '-':<39} {result.detail}")
    passed, failed, skipped = (
        sum(result.status == status for result in results) for status in ("pass", "fail", "skip")
    )
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    if args.output is not None:
        write_report(results, args.output, device=args.device)
    sys.exit(exit_code(results))


if __name__ == "__main__":
    main()
