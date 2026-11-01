import torch

from src.transformer import TransformerBlock


def test_block_preserves_shape_and_causality():
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=6)
    x = torch.randn(1, 4, 6)

    changed = x.clone()
    changed[:, -1, 0] += 10.0

    with torch.no_grad():
        output = block(x)
        changed_output = block(changed)

    assert tuple(output.shape) == (1, 4, 6)
    assert torch.isfinite(output).all().item()
    assert torch.allclose(
        output[:, :-1, :],
        changed_output[:, :-1, :],
        atol=1e-6,
    )


def test_zero_processing_branches_preserve_input():
    block = TransformerBlock(embedding_dim=6)
    x = torch.randn(1, 4, 6)

    with torch.no_grad():
        for parameter in block.attention.parameters():
            parameter.zero_()

        for parameter in block.feed_forward.parameters():
            parameter.zero_()

        output = block(x)

    assert torch.equal(output, x)


def test_gradients_flow_through_block():
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=6)
    x = torch.randn(1, 4, 6, requires_grad=True)

    output = block(x)
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()
    assert x.grad.abs().sum().item() > 0

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()