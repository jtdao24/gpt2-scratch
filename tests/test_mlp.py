import math

import pytest
import torch
from torch import nn

from gpt.config import GPTConfig
from gpt.model import MLP


def small_config() -> GPTConfig:
    return GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )


def test_output_shape() -> None:
    torch.manual_seed(0)
    mlp = MLP(small_config())
    x = torch.randn(2, 5, 32)

    output = mlp(x)

    assert output.shape == (2, 5, 32)


def test_hidden_width_is_four_times_embedding_width() -> None:
    mlp = MLP(small_config())

    assert mlp.c_fc.weight.shape == (128, 32)
    assert mlp.c_proj.weight.shape == (32, 128)


def test_parameter_names_match_gpt2_mlp() -> None:
    mlp = MLP(small_config())

    assert set(dict(mlp.named_parameters())) == {
        "c_fc.weight",
        "c_fc.bias",
        "c_proj.weight",
        "c_proj.bias",
    }


def test_positions_are_transformed_independently() -> None:
    torch.manual_seed(1)
    mlp = MLP(small_config())
    x = torch.randn(1, 5, 32)
    y1 = mlp(x)
    x2 = x.clone()
    x2[:, 2] = torch.randn_like(x2[:, 2])

    y2 = mlp(x2)

    for position in (0, 1, 3, 4):
        assert torch.allclose(y1[0, position], y2[0, position], atol=1e-6)
    assert not torch.allclose(y1[0, 2], y2[0, 2])


def test_gelu_uses_tanh_approximation() -> None:
    mlp = MLP(small_config())
    x = torch.linspace(-4, 4, 1001)
    gelu_tanh = 0.5 * x * (
        1
        + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
    )

    output = mlp.gelu(x)

    assert torch.allclose(output, gelu_tanh, atol=1e-6)
    assert not torch.allclose(output, nn.GELU()(x), atol=1e-5)


def test_matches_independent_tanh_gelu_reference() -> None:
    torch.manual_seed(2)
    mlp = MLP(small_config())
    x = torch.randn(2, 5, 32)
    hidden = mlp.c_fc(x)
    gelu_tanh = 0.5 * hidden * (
        1
        + torch.tanh(
            math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3)
        )
    )
    reference = mlp.c_proj(gelu_tanh)

    assert torch.allclose(mlp(x), reference, atol=1e-6)


def test_output_projection_has_scale_init_marker() -> None:
    mlp = MLP(small_config())

    assert mlp.c_proj.GPT_SCALE_INIT is True


def test_gradients_flow_through_mlp() -> None:
    torch.manual_seed(3)
    mlp = MLP(small_config())
    x = torch.randn(2, 5, 32)

    mlp(x).sum().backward()

    assert mlp.c_fc.weight.grad is not None
    assert mlp.c_proj.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_mlp_runs_on_cuda() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    mlp = MLP(small_config()).to(device)
    x = torch.randn(2, 5, 32, device=device)

    output = mlp(x)

    assert output.device == x.device


def test_default_mlp_parameter_count() -> None:
    mlp = MLP(GPTConfig())

    parameter_count = sum(parameter.numel() for parameter in mlp.parameters())

    assert parameter_count == 4_722_432
