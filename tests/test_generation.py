import pytest
import torch

from src.generation import generate
from src.model import BigramLanguageModel


def test_generation_appends_selected_tokens_without_changing_parameters():
    model = BigramLanguageModel(vocab_size=8, embedding_dim=6)

    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[2] = 10.0

    prompt = torch.tensor([[3, 4]], dtype=torch.long)
    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    result = generate(model, prompt, max_new_tokens=3)

    assert result.tolist() == [[3, 4, 2, 2, 2]]
    assert prompt.tolist() == [[3, 4]]
    assert tuple(result.shape) == (1, 5)

    for before, after in zip(parameters_before, model.parameters()):
        assert torch.equal(before, after)


def test_zero_new_tokens_preserves_prompt():
    model = BigramLanguageModel(vocab_size=8, embedding_dim=6)
    prompt = torch.tensor([[3, 2]], dtype=torch.long)

    result = generate(model, prompt, max_new_tokens=0)

    assert torch.equal(result, prompt)


def test_empty_prompt_fails():
    model = BigramLanguageModel(vocab_size=8, embedding_dim=6)
    prompt = torch.empty((1, 0), dtype=torch.long)

    with pytest.raises(ValueError, match="nonempty sequence"):
        generate(model, prompt, max_new_tokens=3)


def test_negative_token_count_fails():
    model = BigramLanguageModel(vocab_size=8, embedding_dim=6)
    prompt = torch.tensor([[3]], dtype=torch.long)

    with pytest.raises(ValueError, match="must not be negative"):
        generate(model, prompt, max_new_tokens=-1)