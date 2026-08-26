import pytest
import torch

from gpt import GPT, GPTConfig


def small_config() -> GPTConfig:
    return GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )


def test_output_shape() -> None:
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))

    output = model(idx)

    assert output.shape == (2, 5, 100)


def test_output_dtype_is_float32() -> None:
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))

    output = model(idx)

    assert output.dtype == torch.float32


def test_sequence_longer_than_block_size_raises() -> None:
    model = GPT(small_config())
    idx = torch.zeros((1, 17), dtype=torch.long)

    with pytest.raises(ValueError, match=r"T=17.*block_size=16"):
        model(idx)


def test_sequence_equal_to_block_size_is_allowed() -> None:
    model = GPT(small_config())
    idx = torch.zeros((1, 16), dtype=torch.long)

    output = model(idx)

    assert output.shape == (1, 16, 100)


def test_position_changes_embedding_for_same_token() -> None:
    model = GPT(small_config())
    idx = torch.tensor([[7, 7]], dtype=torch.long)

    output = model(idx)

    assert not torch.equal(output[0, 0], output[0, 1])


def test_embedding_parameter_names_match_gpt2_checkpoint() -> None:
    model = GPT(small_config())

    names = set(dict(model.named_parameters()))
    assert {
        "transformer.wte.weight",
        "transformer.wpe.weight",
    } <= names


def test_default_embedding_shapes() -> None:
    model = GPT(GPTConfig())

    assert model.transformer.wte.weight.shape == (50257, 768)
    assert model.transformer.wpe.weight.shape == (1024, 768)


def test_token_embeddings_use_gpt2_initialization() -> None:
    torch.manual_seed(0)
    model = GPT(small_config())

    std = model.transformer.wte.weight.std().item()

    assert 0.015 < std < 0.025


def test_embedding_weights_receive_gradients() -> None:
    model = GPT(small_config())
    idx = torch.tensor([[1, 2, 1]], dtype=torch.long)

    model(idx).sum().backward()

    assert model.transformer.wte.weight.grad is not None
    assert model.transformer.wpe.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_forward_uses_input_device_for_positions() -> None:
    device = torch.device("cuda")
    model = GPT(small_config()).to(device)
    idx = torch.randint(0, 100, (2, 5), device=device)

    output = model(idx)

    assert output.device == idx.device
