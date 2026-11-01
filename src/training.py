import torch
from torch.nn import functional as F

from src.model import BigramLanguageModel


def calculate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


def train_model(
    model: BigramLanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 200,
    learning_rate: float = 0.05,
) -> tuple[float, float]:
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    with torch.no_grad():
        initial_loss = calculate_loss(model(x), y).item()

    for _ in range(steps):
        optimizer.zero_grad()

        logits = model(x)
        loss = calculate_loss(logits, y)

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = calculate_loss(model(x), y).item()

    return initial_loss, final_loss