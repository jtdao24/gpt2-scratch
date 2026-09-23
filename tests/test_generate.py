from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from gpt import GPT, GPTConfig
from gpt.checkpoint import save_checkpoint
from gpt.generate import generate, generate_text, load_model_from_checkpoint


def tiny_config() -> GPTConfig:
    return GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )


def test_output_shape_and_dtype() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    prompt = torch.randint(0, model.config.vocab_size, (2, 5))

    output = generate(model, prompt, max_new_tokens=7)

    assert output.shape == (2, 12)
    assert output.dtype == torch.int64


def test_generated_ids_are_within_vocabulary() -> None:
    torch.manual_seed(1)
    model = GPT(tiny_config())
    prompt = torch.randint(0, model.config.vocab_size, (3, 5))

    output = generate(model, prompt, max_new_tokens=10, top_k=None)

    generated = output[:, prompt.size(1) :]
    assert torch.all(generated >= 0)
    assert torch.all(generated < model.config.vocab_size)


def test_prompt_is_preserved_as_prefix() -> None:
    torch.manual_seed(2)
    model = GPT(tiny_config())
    prompt = torch.randint(0, model.config.vocab_size, (2, 5))

    output = generate(model, prompt, max_new_tokens=6)

    assert torch.equal(output[:, : prompt.size(1)], prompt)


def test_seeded_generators_are_reproducible() -> None:
    torch.manual_seed(3)
    model = GPT(tiny_config())
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    first_generator = torch.Generator().manual_seed(10)
    second_generator = torch.Generator().manual_seed(10)
    different_generator = torch.Generator().manual_seed(11)

    first = generate(
        model,
        prompt,
        max_new_tokens=12,
        top_k=None,
        generator=first_generator,
    )
    second = generate(
        model,
        prompt,
        max_new_tokens=12,
        top_k=None,
        generator=second_generator,
    )
    different = generate(
        model,
        prompt,
        max_new_tokens=12,
        top_k=None,
        generator=different_generator,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_top_k_one_is_greedy() -> None:
    torch.manual_seed(4)
    model = GPT(tiny_config())
    generated = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    for _ in range(3):
        with torch.no_grad():
            logits, _ = model(generated[:, -model.config.block_size :])
            expected = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        next_output = generate(
            model,
            generated,
            max_new_tokens=1,
            top_k=1,
        )
        assert torch.equal(next_output[:, -1:], expected)
        generated = next_output


def test_nonpositive_temperature_is_rejected() -> None:
    model = GPT(tiny_config())
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    for temperature in (0.0, -1.0):
        with pytest.raises(ValueError, match=str(temperature)):
            generate(model, prompt, max_new_tokens=1, temperature=temperature)


def test_context_is_cropped_to_block_size() -> None:
    torch.manual_seed(5)
    model = GPT(tiny_config())
    prompt = torch.randint(0, model.config.vocab_size, (1, 40))

    output = generate(model, prompt, max_new_tokens=5)

    assert output.shape == (1, 45)


def test_training_mode_is_restored() -> None:
    model = GPT(tiny_config())
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    model.train()
    generate(model, prompt, max_new_tokens=1)
    assert model.training is True

    model.eval()
    generate(model, prompt, max_new_tokens=1)
    assert model.training is False


def test_generation_does_not_change_model_or_gradients() -> None:
    torch.manual_seed(6)
    model = GPT(tiny_config())
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()

    state_before = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }
    gradients_before = [parameter.grad for parameter in model.parameters()]
    gradient_values_before = [
        gradient.clone() if gradient is not None else None
        for gradient in gradients_before
    ]

    generate(model, idx[:1, :3], max_new_tokens=5)

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state_before[name])
    for parameter, original_gradient, original_value in zip(
        model.parameters(), gradients_before, gradient_values_before
    ):
        assert parameter.grad is original_gradient
        assert parameter.grad is not None
        assert original_value is not None
        assert torch.equal(parameter.grad, original_value)


def test_generation_continues_learned_pattern() -> None:
    torch.manual_seed(0)
    config = GPTConfig(
        block_size=32,
        vocab_size=20,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sequence = torch.arange(10).repeat(200)
    window_size = config.block_size

    for step in range(300):
        start = (step * 8) % (len(sequence) - window_size - 1)
        window = sequence[start : start + window_size + 1]
        idx = window[:-1].unsqueeze(0)
        targets = window[1:].unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

    prompt = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    output = generate(model, prompt, max_new_tokens=12, top_k=1)

    assert output.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]]


def test_generate_text_round_trip() -> None:
    torch.manual_seed(7)
    config = GPTConfig(
        block_size=32,
        vocab_size=50257,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)
    prompt = "Hello, world"

    samples = generate_text(
        model,
        prompt,
        max_new_tokens=3,
        num_samples=2,
        top_k=1,
    )

    assert len(samples) == 2
    assert all(sample.startswith(prompt) for sample in samples)


def test_generate_text_rejects_empty_prompt() -> None:
    model = GPT(tiny_config())

    with pytest.raises(ValueError, match="prompt"):
        generate_text(model, "", max_new_tokens=1)


def test_load_model_from_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(8)
    config = tiny_config()
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=3,
        train_position=48,
        val_loss=4.0,
        train_config={"device": "cpu"},
        model_config=asdict(config),
    )

    restored = load_model_from_checkpoint(checkpoint_path)

    assert asdict(restored.config) == asdict(config)
    assert restored.training is False
    for name, tensor in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_generate_on_cuda_with_seeded_generator() -> None:
    torch.manual_seed(9)
    model = GPT(tiny_config()).to("cuda")
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(10)

    output = generate(
        model,
        prompt,
        max_new_tokens=3,
        generator=generator,
    )

    assert output.device == prompt.device
