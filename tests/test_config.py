import pytest

from gpt import GPTConfig


def test_default_config() -> None:
    config = GPTConfig()

    assert config.block_size == 1024
    assert config.vocab_size == 50257
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_embd == 768
    assert config.head_dim == 64


def test_embedding_dimension_must_be_divisible_by_heads() -> None:
    with pytest.raises(ValueError):
        GPTConfig(n_embd=768, n_head=7)


def test_layer_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        GPTConfig(n_layer=0)


def test_medium_pretrained_config() -> None:
    config = GPTConfig.from_pretrained_name("gpt2-medium")

    assert config.n_layer == 24
    assert config.n_head == 16
    assert config.n_embd == 1024
    assert config.head_dim == 64


def test_unknown_pretrained_name() -> None:
    with pytest.raises(ValueError):
        GPTConfig.from_pretrained_name("gpt2-tiny")
