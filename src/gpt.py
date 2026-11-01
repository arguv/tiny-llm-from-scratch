import torch
from torch import nn

from src.embeddings import TokenEmbedding
from src.transformer import TransformerBlock


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 6,
        block_size: int = 4,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim,
            device="cpu",
        )
        self.block = TransformerBlock(embedding_dim)
        self.final_norm = nn.LayerNorm(embedding_dim, device="cpu")
        self.output = nn.Linear(
            embedding_dim,
            vocab_size,
            device="cpu",
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence).")

        sequence_length = token_ids.shape[1]

        if not 1 <= sequence_length <= self.block_size:
            raise ValueError("Sequence length must be between 1 and block_size.")

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.block(x)
        x = self.final_norm(x)

        return self.output(x)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("Provide a nonempty sequence with two dimensions.")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative.")

        self.eval()
        generated = token_ids.clone()

        for _ in range(max_new_tokens):
            context = generated[:, -self.block_size:]
            logits = self(context)
            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )
            generated = torch.cat((generated, next_token), dim=1)

        return generated