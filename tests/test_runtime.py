import pytest
import torch

from src.gpt import TinyGPT
from src.runtime import inspect_runtime


def test_runtime_matches_forward_and_preserves_parameters():
    torch.manual_seed(42)
    model = TinyGPT(vocab_size=8)
    prompt = torch.tensor([[3, 2, 4, 4]], dtype=torch.long)

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    trace = inspect_runtime(model, prompt)

    with torch.no_grad():
        expected_logits = model(prompt)

    assert torch.allclose(trace["logits"], expected_logits)
    assert tuple(trace["probabilities"].shape) == (1, 8)
    assert torch.isfinite(trace["probabilities"]).all().item()
    assert (trace["probabilities"] >= 0).all().item()
    assert torch.allclose(
        trace["probabilities"].sum(dim=-1),
        torch.ones(1),
    )
    assert torch.equal(
        trace["next_token"],
        expected_logits[:, -1, :].argmax(dim=-1, keepdim=True),
    )

    for original, current in zip(before, model.parameters()):
        assert torch.equal(original, current)


def test_runtime_uses_last_context_window():
    model = TinyGPT(vocab_size=8, block_size=4)
    token_ids = torch.tensor([[3, 2, 4, 4, 5]], dtype=torch.long)

    trace = inspect_runtime(model, token_ids)

    assert trace["context"].tolist() == [[2, 4, 4, 5]]
    assert trace["positions"].tolist() == [0, 1, 2, 3]
    assert tuple(trace["logits"].shape) == (1, 4, 8)


def test_runtime_rejects_empty_prompt():
    model = TinyGPT(vocab_size=8)
    token_ids = torch.empty((1, 0), dtype=torch.long)

    with pytest.raises(ValueError, match="nonempty sequence"):
        inspect_runtime(model, token_ids)