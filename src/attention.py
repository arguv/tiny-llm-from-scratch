import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim, embedding_dim, bias=False, device="cpu"
        )
        self.key = nn.Linear(
            embedding_dim, embedding_dim, bias=False, device="cpu"
        )
        self.value = nn.Linear(
            embedding_dim, embedding_dim, bias=False, device="cpu"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = x.shape[1]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (q.shape[-1] ** 0.5)

        allowed = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            )
        )
        scores = scores.masked_fill(~allowed, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v

        return output, weights