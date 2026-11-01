import torch

from src.embeddings import TokenEmbedding
from src.tokenizer import CharTokenizer
from src.transformer import TransformerBlock


def main():
    torch.manual_seed(42)

    tokenizer = CharTokenizer("hello world")
    token_ids = torch.tensor(
        [tokenizer.encode("hell")],
        dtype=torch.long,
    )

    embedding = TokenEmbedding(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=6,
    )
    block = TransformerBlock(embedding_dim=6)
    block.eval()

    with torch.no_grad():
        vectors = embedding(token_ids)
        output = block(vectors)

        changed = vectors.clone()
        changed[:, -1, 0] += 10.0
        changed_output = block(changed)

    print("Token IDs shape:", tuple(token_ids.shape))
    print("Block input shape:", tuple(vectors.shape))
    print("Block output shape:", tuple(output.shape))
    print("Output is finite:", torch.isfinite(output).all().item())
    print(
        "Earlier positions unchanged:",
        torch.allclose(
            output[:, :-1, :],
            changed_output[:, :-1, :],
            atol=1e-6,
        ),
    )


if __name__ == "__main__":
    main()