import math

import pytest
import torch
from torch.nn import functional as F

from gpt.config import GPTConfig
from gpt.model import CausalSelfAttention


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
    attn = CausalSelfAttention(small_config())
    x = torch.randn(2, 5, 32)

    output = attn(x)

    assert output.shape == (2, 5, 32)


def test_future_token_does_not_affect_earlier_outputs() -> None:
    torch.manual_seed(1)
    attn = CausalSelfAttention(small_config())
    x = torch.randn(1, 5, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, -1] = torch.randn_like(x2[:, -1])

    y2 = attn(x2)

    assert torch.allclose(y1[0, :-1], y2[0, :-1], atol=1e-6)
    assert not torch.allclose(y1[0, -1], y2[0, -1])


def test_first_token_affects_every_output_position() -> None:
    torch.manual_seed(2)
    attn = CausalSelfAttention(small_config())
    x = torch.randn(1, 5, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 0] = torch.randn_like(x2[:, 0])

    y2 = attn(x2)

    for position in range(x.size(1)):
        assert not torch.allclose(y1[0, position], y2[0, position])


def test_attention_rows_sum_to_one() -> None:
    torch.manual_seed(3)
    attn = CausalSelfAttention(small_config())
    x = torch.randn(2, 5, 32)
    B, T, C = x.size()
    q, k, _ = attn.c_attn(x).split(attn.n_embd, dim=2)
    q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    weights = weights.masked_fill(~attn.bias[:, :, :T, :T], float("-inf"))
    weights = F.softmax(weights, dim=-1)

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_future_attention_weights_are_zero() -> None:
    torch.manual_seed(4)
    attn = CausalSelfAttention(small_config())
    x = torch.randn(2, 5, 32)
    B, T, C = x.size()
    q, k, _ = attn.c_attn(x).split(attn.n_embd, dim=2)
    q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    weights = weights.masked_fill(~attn.bias[:, :, :T, :T], float("-inf"))
    weights = F.softmax(weights, dim=-1)
    upper_triangle = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    assert torch.count_nonzero(weights[..., upper_triangle]).item() == 0


def test_head_split_uses_contiguous_channel_slices() -> None:
    config = small_config()
    B, T, C = 1, 2, config.n_embd
    hs = config.head_dim
    channels = torch.arange(C).view(1, 1, C).expand(B, T, C)

    heads = channels.view(B, T, config.n_head, hs).transpose(1, 2)

    for head in range(config.n_head):
        expected = torch.arange(head * hs, (head + 1) * hs).expand(B, T, hs)
        assert torch.equal(heads[:, head], expected)


def test_short_sequences_work() -> None:
    torch.manual_seed(5)
    attn = CausalSelfAttention(small_config())
    x1 = torch.randn(2, 1, 32)
    x3 = torch.randn(2, 3, 32)

    output1 = attn(x1)
    output3 = attn(x3)

    assert output1.shape == (2, 1, 32)
    assert output3.shape == (2, 3, 32)

    B, T, C = x1.size()
    q, k, _ = attn.c_attn(x1).split(attn.n_embd, dim=2)
    q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    weights = weights.masked_fill(~attn.bias[:, :, :T, :T], float("-inf"))
    weights = F.softmax(weights, dim=-1)

    assert torch.equal(weights, torch.ones_like(weights))


def test_parameter_names_match_gpt2_attention() -> None:
    attn = CausalSelfAttention(small_config())

    assert set(dict(attn.named_parameters())) == {
        "c_attn.weight",
        "c_attn.bias",
        "c_proj.weight",
        "c_proj.bias",
    }


def test_causal_mask_is_not_in_state_dict() -> None:
    attn = CausalSelfAttention(small_config())

    assert "bias" not in attn.state_dict()


def test_gradients_flow_through_attention() -> None:
    torch.manual_seed(6)
    attn = CausalSelfAttention(small_config())
    x = torch.randn(2, 5, 32)

    attn(x).sum().backward()

    assert attn.c_attn.weight.grad is not None
    assert attn.c_proj.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_attention_runs_on_cuda() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    attn = CausalSelfAttention(small_config()).to(device)
    x = torch.randn(2, 5, 32, device=device)

    output = attn(x)

    assert output.device == x.device


def test_matches_independent_scaled_dot_product_reference() -> None:
    torch.manual_seed(8)
    attn = CausalSelfAttention(small_config())
    x = torch.randn(2, 5, 32)
    q, k, v = attn.c_attn(x).split(attn.n_embd, dim=2)
    hs = attn.n_embd // attn.n_head
    head_outputs = []

    for head in range(attn.n_head):
        channel_slice = slice(head * hs, (head + 1) * hs)
        qh = q[:, :, channel_slice]
        kh = k[:, :, channel_slice]
        vh = v[:, :, channel_slice]
        head_outputs.append(
            F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        )

    reference = attn.c_proj(torch.cat(head_outputs, dim=2))

    assert torch.allclose(attn(x), reference, atol=1e-5)
