"""Verify the GPT-2 harness with real models and deliberate parameter faults."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gpt import GPT, GPTConfig
from gpt.hf_weights import from_pretrained
from gpt.validate import (
    CheckResult,
    aligned_loss,
    check_gpu,
    check_text_loss,
    compare_logits,
    exit_code,
    load_reference,
    run_validation,
    write_report,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def pretrained_models() -> tuple[GPT, nn.Module]:
    return from_pretrained("gpt2", "cpu"), load_reference()


def test_compare_logits() -> None:
    ref = torch.tensor([[[1.0, 0.75], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]])
    assert compare_logits(ref, ref) == {
        "max_abs_diff": 0.0,
        "relative_diff": 0.0,
        "argmax_agreement": 1.0,
    }
    ours = ref.clone()
    ours[0, 0, 1] += 0.5
    assert compare_logits(ours, ref) == {
        "max_abs_diff": 0.5,
        "relative_diff": 0.25,
        "argmax_agreement": 0.75,
    }


def test_aligned_loss_scores_shifted_pairs() -> None:
    torch.manual_seed(0)
    config = GPTConfig(block_size=16, vocab_size=50, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (2, 9))

    with torch.no_grad():
        logits, _ = model(idx[:, :-1])
        expected = F.cross_entropy(logits.flatten(0, 1), idx[:, 1:].reshape(-1))
        actual = aligned_loss(model, idx)
        _, unshifted = model(idx, targets=idx)

    assert torch.equal(actual, expected)
    assert unshifted is not None
    assert not torch.isclose(actual, unshifted)


def test_exit_code() -> None:
    results = [CheckResult(status, status, None, "", "") for status in ("pass", "skip", "info")]
    assert exit_code(results) == 0
    assert exit_code([*results, CheckResult("bad", "fail", 2.0, "< 1", "")]) == 1


def test_write_report_round_trip(tmp_path: Path) -> None:
    results = [
        CheckResult("parameters", "pass", 0.0, "0 mismatches", "all tensors match"),
        CheckResult("text loss", "skip", None, "< 1e-5", "validation split missing"),
    ]
    path = tmp_path / "reports" / "validation.json"
    write_report(results, path)
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report.keys() == {
        "created", "torch", "transformers", "python", "device", "git_commit", "checks"
    }
    assert datetime.fromisoformat(report["created"]).utcoffset() == timedelta(0)
    assert report["device"] == "cpu"
    assert report["checks"] == [asdict(result) for result in results]


@pytest.mark.network
def test_run_validation_cpu(pretrained_models: tuple[GPT, nn.Module]) -> None:
    ours, ref = pretrained_models
    results = run_validation("cpu", DATA_DIR, ours=ours, ref=ref)
    by_name = {result.name: result.status for result in results}

    assert not [result for result in results if result.status == "fail"]
    for name in (
        "architecture", "parameters", "aligned loss", "eager reference",
        "logits (1, 1)", "logits (1, 16)", "logits (2, 64)", "logits (4, 256)",
        "logits (2, 1024)", "generation: The meaning of life is",
        "generation: Hello, I am a language model,", "generation: In 1492, Columbus",
    ):
        assert by_name[name] == "pass"


@pytest.mark.network
def test_validation_detects_wrong_transpose(
    pretrained_models: tuple[GPT, nn.Module], tmp_path: Path
) -> None:
    ours, ref = pretrained_models
    broken = deepcopy(ours)
    with torch.no_grad():
        weight = broken.transformer.h[0].attn.c_proj.weight
        weight.copy_(weight.t().contiguous())

    results = run_validation("cpu", tmp_path, ours=broken, ref=ref)
    parameters = next(result for result in results if result.name == "parameters")
    logits = [result for result in results if result.name.startswith("logits ")]
    assert parameters.status == "fail"
    assert parameters.value == 1.0
    assert len(logits) == 5
    assert all(result.status == "fail" for result in logits)


@pytest.mark.network
def test_validation_detects_value_only_fault(
    pretrained_models: tuple[GPT, nn.Module], tmp_path: Path
) -> None:
    ours, ref = pretrained_models
    broken = deepcopy(ours)
    with torch.no_grad():
        broken.transformer.h[5].ln_2.bias.add_(0.01)

    results = run_validation("cpu", tmp_path, ours=broken, ref=ref)
    logits = [result for result in results if result.name.startswith("logits ")]
    assert len(logits) == 5
    assert all(result.status == "fail" for result in logits)


@pytest.mark.network
@pytest.mark.skipif(not (DATA_DIR / "val.bin").is_file(), reason="val.bin missing")
def test_text_loss(pretrained_models: tuple[GPT, nn.Module]) -> None:
    ours, ref = pretrained_models
    result = check_text_loss(ours, ref, DATA_DIR, "cpu")

    assert result.status == "pass", result.detail
    assert result.value is not None
    assert 4.0 < result.value < 4.5


@pytest.mark.network
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_checks(pretrained_models: tuple[GPT, nn.Module]) -> None:
    ours, ref = pretrained_models
    results = check_gpu(ours, ref)

    assert not [result for result in results if result.status == "fail"]
    assert results[0].status == "pass"
    assert next(ours.parameters()).device.type == "cpu"
    assert next(ref.parameters()).device.type == "cpu"
