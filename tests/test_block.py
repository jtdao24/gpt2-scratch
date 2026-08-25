import pytest
import torch

from gpt.config import GPTConfig
from gpt.model import Block


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
    block = Block(small_config())
    x = torch.randn(2, 5, 32)

    output = block(x)

    assert output.shape == (2, 5, 32)


def test_zeroed_branches_make_block_an_identity() -> None:
    torch.manual_seed(1)
    block = Block(small_config())
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        for layer in (block.attn.c_proj, block.mlp.c_proj):
            layer.weight.zero_()
            layer.bias.zero_()

    output = block(x)

    assert torch.equal(output, x)


def test_block_uses_pre_layer_norm() -> None:
    torch.manual_seed(2)
    block = Block(small_config())
    x = torch.randn(1, 5, 32) * 100

    output = block(x)

    assert output.std().item() > 50


def test_matches_independent_pre_layer_norm_reference() -> None:
    torch.manual_seed(3)
    block = Block(small_config())
    x = torch.randn(2, 5, 32)
    h = x + block.attn(block.ln_1(x))
    reference = h + block.mlp(block.ln_2(h))

    assert torch.allclose(block(x), reference, atol=1e-6)


def test_causality_survives_the_block() -> None:
    torch.manual_seed(4)
    block = Block(small_config())
    x = torch.randn(1, 5, 32)
    y1 = block(x)
    x2 = x.clone()
    x2[:, -1] = torch.randn_like(x2[:, -1])

    y2 = block(x2)

    assert torch.allclose(y1[0, :-1], y2[0, :-1], atol=1e-6)
    assert not torch.allclose(y1[0, -1], y2[0, -1])


def test_layer_norms_are_distinct_modules() -> None:
    block = Block(small_config())

    assert block.ln_1 is not block.ln_2


def test_parameter_names_match_gpt2_block() -> None:
    block = Block(small_config())

    assert set(dict(block.named_parameters())) == {
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
    }


def test_layer_norm_configuration_matches_gpt2() -> None:
    block = Block(small_config())

    assert block.ln_1.eps == 1e-5
    assert block.ln_1.weight is not None
    assert block.ln_1.bias is not None
    assert torch.equal(block.ln_1.weight, torch.ones_like(block.ln_1.weight))
    assert torch.equal(block.ln_1.bias, torch.zeros_like(block.ln_1.bias))


def test_default_block_parameter_count() -> None:
    block = Block(GPTConfig())

    parameter_count = sum(parameter.numel() for parameter in block.parameters())

    assert parameter_count == 7_087_872


def test_gradients_flow_to_every_submodule() -> None:
    torch.manual_seed(5)
    block = Block(small_config())
    x = torch.randn(2, 5, 32)

    block(x).sum().backward()

    assert block.ln_1.weight.grad is not None
    assert block.attn.c_attn.weight.grad is not None
    assert block.ln_2.weight.grad is not None
    assert block.mlp.c_fc.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_block_runs_on_cuda() -> None:
    torch.manual_seed(6)
    device = torch.device("cuda")
    block = Block(small_config()).to(device)
    x = torch.randn(2, 5, 32, device=device)

    output = block(x)

    assert output.device == x.device
