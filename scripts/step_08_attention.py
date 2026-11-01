import torch

from src.attention import CausalSelfAttention
from src.embeddings import TokenEmbedding
from src.tokenizer import CharTokenizer


def main():
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    tokenizer = CharTokenizer("hello world")
    token_ids = torch.tensor(
        [tokenizer.encode("hell")],
        dtype=torch.long,
    )

    embedding = TokenEmbedding(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=6,
    )
    attention = CausalSelfAttention(embedding_dim=6)

    with torch.no_grad():
        attention.query.weight.zero_()
        attention.key.weight.zero_()

        vectors = embedding(token_ids)
        output, weights = attention(vectors)

    print("Text:", tokenizer.decode(token_ids[0].tolist()))
    print("Token IDs shape:", tuple(token_ids.shape))
    print("Embedding shape:", tuple(vectors.shape))
    print("Output shape:", tuple(output.shape))
    print("Attention weights shape:", tuple(weights.shape))

    print("\nAttention weights:")
    print(weights[0])

    print(
        "\nRows sum to one:",
        torch.allclose(
            weights.sum(dim=-1),
            torch.ones_like(weights.sum(dim=-1)),
        ),
    )
    print(
        "Future positions have zero weight:",
        torch.count_nonzero(torch.triu(weights, diagonal=1)).item() == 0,
    )


if __name__ == "__main__":
    main()