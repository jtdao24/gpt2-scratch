"""Pretrained weight conversion and integration tests for our GPT model."""

from pathlib import Path

import pytest
import torch

from gpt import GPT, GPTConfig
from gpt.checkpoint import load_checkpoint
from gpt.generate import generate, load_model_from_checkpoint
from gpt.hf_weights import (
    HF_MODEL_IDS,
    TRANSPOSED_SUFFIXES,
    convert_state_dict,
    from_pretrained,
    main,
)


@pytest.fixture(scope="module")
def pretrained_models() -> tuple[GPT, torch.nn.Module]:
    """Load each implementation once, reusing Hugging Face's download cache."""
    from transformers import GPT2LMHeadModel

    reference = GPT2LMHeadModel.from_pretrained(HF_MODEL_IDS["gpt2"]).eval()
    return from_pretrained("gpt2"), reference


def test_convert_state_dict() -> None:
    transposed = {
        "transformer.h.0.attn.c_attn.weight": torch.arange(12).view(2, 6).float(),
        "transformer.h.0.attn.c_proj.weight": torch.arange(4).view(2, 2).float(),
        "transformer.h.0.mlp.c_fc.weight": torch.arange(16).view(2, 8).float(),
        "transformer.h.0.mlp.c_proj.weight": torch.arange(16).view(8, 2).float(),
    }
    unchanged = {
        "transformer.wte.weight": torch.arange(8).view(4, 2).float(),
        "transformer.h.0.ln_1.weight": torch.ones(2),
        "transformer.h.0.attn.c_attn.bias": torch.arange(6).float(),
    }
    original = {**transposed, **unchanged}
    before = {key: value.clone() for key, value in original.items()}

    converted = convert_state_dict(original)

    assert converted is not original
    assert set(converted) == set(original)
    for key, value in transposed.items():
        assert torch.equal(converted[key], value.t())
    for key, value in unchanged.items():
        assert converted[key] is value
    for key, value in original.items():
        assert torch.equal(value, before[key])
        assert converted[key].is_contiguous()


def test_square_attention_projection_is_transposed() -> None:
    key = "transformer.h.0.attn.c_proj.weight"
    weight = torch.arange(768 * 768).float().view(768, 768)

    converted = convert_state_dict({key: weight})[key]

    assert converted.shape == weight.shape
    assert torch.equal(converted, weight.t())
    assert not torch.equal(converted, weight)
    assert converted.is_contiguous()


@pytest.mark.network
def test_pretrained_model_parameters_mode_and_tying(
    pretrained_models: tuple[GPT, torch.nn.Module],
) -> None:
    model, _ = pretrained_models

    assert isinstance(model, GPT)
    assert sum(parameter.numel() for parameter in model.parameters()) == 124_439_808
    assert model.training is False
    assert model.lm_head.weight is model.transformer.wte.weight


@pytest.mark.network
def test_converted_keys_exactly_match_our_model(
    pretrained_models: tuple[GPT, torch.nn.Module],
) -> None:
    model, reference = pretrained_models
    converted = convert_state_dict(reference.state_dict())
    our_keys = set(model.state_dict())

    assert len(converted) == 149
    assert len(our_keys) == 149
    assert set(converted) == our_keys


@pytest.mark.network
def test_pretrained_logits_match_reference(
    pretrained_models: tuple[GPT, torch.nn.Module],
) -> None:
    model, reference = pretrained_models
    generator = torch.Generator().manual_seed(1337)
    idx = torch.randint(
        0, model.config.vocab_size, (1, 16), dtype=torch.int64, generator=generator
    )

    with torch.no_grad():
        ours, _ = model(idx)
        expected = reference(idx).logits

    maximum_difference = (ours - expected).abs().max().item()
    assert maximum_difference < 1e-4, maximum_difference


@pytest.mark.network
def test_omitting_square_transpose_breaks_logits(
    pretrained_models: tuple[GPT, torch.nn.Module],
) -> None:
    _, reference = pretrained_models
    incomplete_suffixes = tuple(
        suffix
        for suffix in TRANSPOSED_SUFFIXES
        if suffix != "attn.c_proj.weight"
    )
    incomplete_state = {
        key: value.t().contiguous()
        if key.endswith(incomplete_suffixes)
        else value
        for key, value in reference.state_dict().items()
    }
    broken = GPT(GPTConfig.from_pretrained_name("gpt2"))
    broken.load_state_dict(incomplete_state)
    broken.eval()
    generator = torch.Generator().manual_seed(1337)
    idx = torch.randint(
        0, broken.config.vocab_size, (1, 16), dtype=torch.int64, generator=generator
    )

    with torch.no_grad():
        actual, _ = broken(idx)
        expected = reference(idx).logits

    assert torch.isfinite(actual).all()
    maximum_difference = (actual - expected).abs().max().item()
    assert maximum_difference > 1.0, maximum_difference


def test_unknown_pretrained_name_is_rejected() -> None:
    with pytest.raises(ValueError) as error:
        from_pretrained("gpt2-tiny")

    message = str(error.value)
    assert "gpt2-tiny" in message
    for name in HF_MODEL_IDS:
        assert name in message


@pytest.mark.network
def test_export_reload_and_generate(
    pretrained_models: tuple[GPT, torch.nn.Module],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, _ = pretrained_models
    path = tmp_path / "pretrained" / "gpt2.pt"

    main(["--model", "gpt2", "--out", str(path), "--device", "cpu"])

    output = capsys.readouterr().out
    assert str(path) in output
    assert "124,439,808" in output
    checkpoint = load_checkpoint(path)
    assert checkpoint["step"] == 0
    assert checkpoint["train_position"] == 0
    assert checkpoint["optimizer"]["state"] == {}
    assert checkpoint["train_config"] == {
        "source": "huggingface",
        "model_type": "gpt2",
    }
    assert checkpoint["val_loss"] != checkpoint["val_loss"]
    del checkpoint

    restored = load_model_from_checkpoint(path)
    assert isinstance(restored, GPT)
    assert restored.training is False
    assert restored.lm_head.weight is restored.transformer.wte.weight
    prompt = torch.tensor([[464, 3616, 286, 1204, 318]], dtype=torch.int64)
    generated = generate(restored, prompt, max_new_tokens=5, top_k=1)

    assert generated.shape == (1, prompt.size(1) + 5)
    assert torch.equal(generated[:, : prompt.size(1)], prompt)
    assert torch.all(generated >= 0)
    assert torch.all(generated < model.config.vocab_size)
