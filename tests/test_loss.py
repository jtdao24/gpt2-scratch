import math

import pytest
import torch

from gpt import GPT, GPTConfig


def small_config() -> GPTConfig:
    return GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )


def test_forward_without_targets_returns_no_loss() -> None:
    torch.manual_seed(0)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))

    logits, loss = model(idx)

    assert loss is None
    assert logits.shape == (2, 5, 100)


def test_forward_with_targets_returns_scalar_finite_loss() -> None:
    torch.manual_seed(1)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))
    targets = torch.randint(0, 100, (2, 5))

    _, loss = model(idx, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert math.isfinite(loss.item())


def test_untrained_loss_tracks_small_vocabulary_size() -> None:
    torch.manual_seed(2)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))
    targets = torch.randint(0, 100, (2, 5))

    _, loss = model(idx, targets)

    assert loss is not None
    assert 4.4 < loss.item() < 4.8


def test_untrained_loss_tracks_full_vocabulary_size() -> None:
    torch.manual_seed(3)
    config = GPTConfig()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    targets = torch.randint(0, config.vocab_size, (1, 4))

    _, loss = model(idx, targets)

    assert loss is not None
    assert 10.5 < loss.item() < 11.3


def test_loss_matches_independent_aligned_reference() -> None:
    torch.manual_seed(4)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))
    targets = torch.randint(0, 100, (2, 5))

    logits, loss = model(idx, targets)
    log_probs = torch.log_softmax(logits, dim=-1)
    picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    reference = -picked.mean()

    assert loss is not None
    assert torch.allclose(loss, reference, atol=1e-6)


def test_loss_uses_mean_reduction() -> None:
    torch.manual_seed(5)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (1, 5))
    targets = torch.randint(0, 100, (1, 5))

    _, loss_1 = model(idx, targets)
    _, loss_4 = model(idx.repeat(4, 1), targets.repeat(4, 1))

    assert loss_1 is not None
    assert loss_4 is not None
    assert torch.allclose(loss_1, loss_4, atol=1e-5)


def test_better_predictions_have_lower_loss() -> None:
    torch.manual_seed(6)
    config = small_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 5))
    random_targets = torch.randint(0, config.vocab_size, (2, 5))

    logits, _ = model(idx)
    _, loss_random = model(idx, random_targets)
    easy_targets = logits.argmax(dim=-1)
    _, loss_easy = model(idx, easy_targets)

    assert loss_random is not None
    assert loss_easy is not None
    assert loss_easy.item() < loss_random.item()
    assert loss_easy.item() < math.log(config.vocab_size)


def test_gradients_flow_from_loss() -> None:
    torch.manual_seed(7)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))
    targets = torch.randint(0, 100, (2, 5))

    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()

    assert model.transformer.wte.weight.grad is not None
    assert model.transformer.h[0].attn.c_attn.weight.grad is not None
    assert model.transformer.ln_f.weight.grad is not None


def test_mismatched_target_shape_is_rejected() -> None:
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))
    targets = torch.randint(0, 100, (2, 4))

    with pytest.raises((RuntimeError, ValueError)):
        model(idx, targets)


def test_transposed_target_shape_is_rejected() -> None:
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))
    targets = torch.randint(0, 100, (5, 2))

    with pytest.raises(ValueError, match=r"targets shape.*must match.*idx shape"):
        model(idx, targets)


def test_targets_do_not_change_logits() -> None:
    torch.manual_seed(8)
    model = GPT(small_config())
    idx = torch.randint(0, 100, (2, 5))
    targets = torch.randint(0, 100, (2, 5))

    logits_without_targets, _ = model(idx)
    logits_with_targets, _ = model(idx, targets)

    assert torch.equal(logits_without_targets, logits_with_targets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_loss_runs_on_cuda() -> None:
    torch.manual_seed(9)
    device = torch.device("cuda")
    model = GPT(small_config()).to(device)
    idx = torch.randint(0, 100, (2, 5), device=device)
    targets = torch.randint(0, 100, (2, 5), device=device)

    _, loss = model(idx, targets)

    assert loss is not None
    assert loss.device == idx.device
