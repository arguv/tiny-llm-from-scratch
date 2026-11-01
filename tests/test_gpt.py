import torch

from src.gpt import TinyGPT


def test_gpt_preserves_causality():
    torch.manual_seed(42)
    model = TinyGPT(vocab_size=8)
    original = torch.tensor([[3, 2, 4, 4]], dtype=torch.long)
    changed = torch.tensor([[3, 2, 4, 5]], dtype=torch.long)

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    assert tuple(original_logits.shape) == (1, 4, 8)
    assert torch.isfinite(original_logits).all().item()
    assert torch.allclose(
        original_logits[:, :3, :],
        changed_logits[:, :3, :],
        atol=1e-6,
    )


def test_position_embeddings_affect_output():
    torch.manual_seed(42)
    model = TinyGPT(vocab_size=8)
    token_ids = torch.tensor([[4, 4]], dtype=torch.long)

    with torch.no_grad():
        before = model(token_ids)
        model.position_embedding.weight[1, 0] += 10.0
        after = model(token_ids)

    assert torch.allclose(before[:, 0, :], after[:, 0, :])
    assert not torch.allclose(before[:, 1, :], after[:, 1, :])


def test_generation_crops_context_and_preserves_prompt():
    model = TinyGPT(vocab_size=8, block_size=4)
    prompt = torch.tensor([[3, 2, 4, 4, 5]], dtype=torch.long)

    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[2] = 10.0

    result = model.generate(prompt, max_new_tokens=6)

    assert tuple(result.shape) == (1, 11)
    assert result.tolist() == [[3, 2, 4, 4, 5, 2, 2, 2, 2, 2, 2]]
    assert prompt.tolist() == [[3, 2, 4, 4, 5]]