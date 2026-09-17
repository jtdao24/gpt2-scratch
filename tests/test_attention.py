import math
from unittest import mock

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


def manual_attention_weights(
    attn: CausalSelfAttention, x: torch.Tensor
) -> torch.Tensor:
    B, T, C = x.size()
    q, k, _ = attn.c_attn(x).split(attn.n_embd, dim=2)
    q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    weights = weights.masked_fill(~mask, float("-inf"))
    return F.softmax(weights, dim=-1)


def manual_attention(attn: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.size()
    _, _, v = attn.c_attn(x).split(attn.n_embd, dim=2)
    v = v.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    y = manual_attention_weights(attn, x) @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return attn.c_proj(y)


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
    weights = manual_attention_weights(attn, x)

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_future_attention_weights_are_zero() -> None:
    torch.manual_seed(4)
    attn = CausalSelfAttention(small_config())
    x = torch.randn(2, 5, 32)
    T = x.size(1)
    weights = manual_attention_weights(attn, x)
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

    weights = manual_attention_weights(attn, x1)

    assert torch.equal(weights, torch.ones_like(weights))


def test_parameter_names_match_gpt2_attention() -> None:
    attn = CausalSelfAttention(small_config())

    assert set(dict(attn.named_parameters())) == {
        "c_attn.weight",
        "c_attn.bias",
        "c_proj.weight",
        "c_proj.bias",
    }


def test_attention_has_no_buffers() -> None:
    attn = CausalSelfAttention(small_config())

    assert "bias" not in attn.state_dict()
    assert list(attn.buffers()) == []


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


def test_matches_manual_reference() -> None:
    torch.manual_seed(9)
    attn = CausalSelfAttention(small_config())

    for sequence_length in (1, 3, 5, 16):
        x = torch.randn(2, sequence_length, 32)
        assert torch.allclose(attn(x), manual_attention(attn, x), atol=1e-5)


def test_forward_uses_sdpa() -> None:
    attn = CausalSelfAttention(small_config())
    x = torch.randn(2, 5, 32)

    with mock.patch(
        "torch.nn.functional.scaled_dot_product_attention",
        wraps=F.scaled_dot_product_attention,
    ) as spy:
        attn(x)

    assert spy.call_count == 1
    assert spy.call_args.kwargs.get("is_causal") is True
    assert "attn_mask" not in spy.call_args.kwargs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_full_size_matches_manual_reference_on_cuda() -> None:
    torch.manual_seed(10)
    device = torch.device("cuda")
    attn = CausalSelfAttention(GPTConfig()).to(device)
    x = torch.randn(2, 1024, 768, device=device)

    with torch.no_grad():
        output = attn(x)
        reference = manual_attention(attn, x)

    assert torch.allclose(output, reference, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_sdpa_uses_less_memory_on_cuda() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    attn = CausalSelfAttention(GPTConfig()).to(device)
    x = torch.randn(1, 1024, 768, device=device, requires_grad=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sdpa_baseline = torch.cuda.memory_allocated()
    attn(x).sum().backward()
    torch.cuda.synchronize()
    sdpa_peak = torch.cuda.max_memory_allocated() - sdpa_baseline

    attn.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    manual_baseline = torch.cuda.memory_allocated()
    manual_attention(attn, x).sum().backward()
    torch.cuda.synchronize()
    manual_peak = torch.cuda.max_memory_allocated() - manual_baseline

    assert sdpa_peak < 0.5 * manual_peak
