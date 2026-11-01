import torch
from torch import nn

from src.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim, device="cpu")
        self.attention = CausalSelfAttention(embedding_dim)

        self.norm2 = nn.LayerNorm(embedding_dim, device="cpu")
        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_dim,
                4 * embedding_dim,
                device="cpu",
            ),
            nn.GELU(),
            nn.Linear(
                4 * embedding_dim,
                embedding_dim,
                device="cpu",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.attention(self.norm1(x))
        x = x + attention_output

        feed_forward_output = self.feed_forward(self.norm2(x))
        x = x + feed_forward_output

        return x