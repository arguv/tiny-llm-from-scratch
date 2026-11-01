import torch
from torch import nn

from src.embeddings import TokenEmbedding


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.output = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
            device="cpu",
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.embedding(token_ids)
        logits = self.output(vectors)

        return logits