import torch

from src.attention import CausalSelfAttention


def test_attention_blocks_future_information():
    torch.manual_seed(42)
    attention = CausalSelfAttention(embedding_dim=6)
    x = torch.randn(1, 4, 6)

    changed = x.clone()
    changed[:, 3, :] += 10.0

    with torch.no_grad():
        original_output, weights = attention(x)
        changed_output, _ = attention(changed)

    assert tuple(original_output.shape) == (1, 4, 6)
    assert tuple(weights.shape) == (1, 4, 4)
    assert torch.allclose(
        original_output[:, :3, :],
        changed_output[:, :3, :],
    )
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)).item() == 0
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
    )


def test_uniform_attention_averages_allowed_values():
    torch.manual_seed(42)
    attention = CausalSelfAttention(embedding_dim=6)
    x = torch.randn(1, 4, 6)

    with torch.no_grad():
        attention.query.weight.zero_()
        attention.key.weight.zero_()

        output, _ = attention(x)
        values = attention.value(x)

    for position in range(4):
        expected = values[:, : position + 1, :].mean(dim=1)
        assert torch.allclose(
            output[:, position, :],
            expected,
            atol=1e-6,
        )