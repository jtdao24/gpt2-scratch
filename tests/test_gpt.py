import pytest
import torch
from torch import nn

from gpt import GPT, GPTConfig
from gpt.model import Block


def small_config() -> GPTConfig:
    return GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )


@pytest.fixture(scope="module")
def full_model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig())


def test_logits_shape_and_dtype() -> None:
    torch.manual_seed(1)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))

    logits, _ = model(idx)

    assert logits.shape == (2, 5, 100)
    assert logits.dtype == torch.float32


def test_token_embedding_and_lm_head_weights_are_tied() -> None:
    model = GPT(small_config())

    assert model.lm_head.weight is model.transformer.wte.weight


def test_tied_weight_parameter_and_state_dict_behaviors() -> None:
    model = GPT(small_config())
    parameter_names = set(dict(model.named_parameters()))

    assert "transformer.wte.weight" in parameter_names
    assert "lm_head.weight" not in parameter_names

    state = model.state_dict()
    assert "transformer.wte.weight" in state
    assert "lm_head.weight" in state
    assert state["lm_head.weight"].data_ptr() == state[
        "transformer.wte.weight"
    ].data_ptr()


def test_default_model_parameter_count(full_model: GPT) -> None:
    parameter_count = sum(parameter.numel() for parameter in full_model.parameters())

    # An untied model would have 163,037,184 parameters.
    assert parameter_count == 124_439_808


def test_lm_head_has_no_bias() -> None:
    model = GPT(small_config())

    assert model.lm_head.bias is None


def test_model_has_configured_number_of_blocks() -> None:
    config = small_config()
    model = GPT(config)

    assert len(model.transformer.h) == config.n_layer
    assert all(isinstance(block, Block) for block in model.transformer.h)


def test_default_model_initialization(full_model: GPT) -> None:
    residual_projections = []
    unscaled_projections = [full_model.lm_head]
    for block in full_model.transformer.h:
        residual_projections.extend((block.attn.c_proj, block.mlp.c_proj))
        unscaled_projections.extend((block.attn.c_attn, block.mlp.c_fc))

    assert len(residual_projections) == 24
    for projection in residual_projections:
        assert 0.0035 < projection.weight.std().item() < 0.0047
    for projection in unscaled_projections:
        assert 0.018 < projection.weight.std().item() < 0.022

    for module in full_model.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            assert torch.equal(module.bias, torch.zeros_like(module.bias))

    assert torch.equal(
        full_model.transformer.ln_f.weight,
        torch.ones_like(full_model.transformer.ln_f.weight),
    )
    assert torch.equal(
        full_model.transformer.ln_f.bias,
        torch.zeros_like(full_model.transformer.ln_f.bias),
    )


def test_default_state_dict_keys_match_gpt2(full_model: GPT) -> None:
    expected = {
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "lm_head.weight",
    }
    block_suffixes = (
        "ln_1.weight",
        "ln_1.bias",
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    for index in range(12):
        expected.update(
            f"transformer.h.{index}.{suffix}" for suffix in block_suffixes
        )

    state_keys = set(full_model.state_dict())

    assert state_keys == expected
    assert len(state_keys) == 149


def test_causality_holds_end_to_end() -> None:
    torch.manual_seed(2)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (1, 5))
    logits1, _ = model(idx)
    idx2 = idx.clone()
    idx2[:, -1] = (idx2[:, -1] + 1) % model.config.vocab_size

    logits2, _ = model(idx2)

    assert torch.allclose(logits1[0, :-1], logits2[0, :-1], atol=1e-5)
    assert not torch.allclose(logits1[0, -1], logits2[0, -1])


def test_sequence_length_limits() -> None:
    model = GPT(small_config())
    at_limit = torch.zeros((1, 16), dtype=torch.long)
    over_limit = torch.zeros((1, 17), dtype=torch.long)

    logits, _ = model(at_limit)

    assert logits.shape == (1, 16, 100)
    with pytest.raises(ValueError, match=r"T=17.*block_size=16"):
        model(over_limit)


def test_gradients_flow_end_to_end() -> None:
    torch.manual_seed(3)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))

    model(idx)[0].sum().backward()

    assert model.transformer.wte.weight.grad is not None
    assert model.transformer.h[0].attn.c_attn.weight.grad is not None
    assert model.transformer.ln_f.weight.grad is not None


def test_tied_weight_gradient_is_nonzero() -> None:
    torch.manual_seed(4)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))

    model(idx)[0].sum().backward()

    gradient = model.transformer.wte.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_model_runs_on_cuda() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    model = GPT(small_config()).to(device)
    idx = torch.randint(0, 100, (2, 5), device=device)

    logits, _ = model(idx)

    assert logits.device == idx.device
