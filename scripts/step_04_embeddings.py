import torch

from src.dataset import create_examples
from src.embeddings import TokenEmbedding
from src.tensors import examples_to_tensors
from src.tokenizer import CharTokenizer


def main():
    torch.manual_seed(42)

    tokenizer = CharTokenizer("hello world")
    token_ids = tokenizer.encode("hello world")
    examples = create_examples(token_ids, block_size=4)
    x, y = examples_to_tensors(examples)

    embedding = TokenEmbedding(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=6,
    )

    vectors = embedding(x)

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Embedding table shape:", tuple(embedding.table.weight.shape))
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(vectors.shape))
    print("Output data type:", vectors.dtype)
    print("Device:", vectors.device)

    print("\nFirst input IDs:", x[0].tolist())
    print("First input embeddings shape:", tuple(vectors[0].shape))
    print("One token vector shape:", tuple(vectors[0, 0].shape))

    print(
        "First token retrieves row 3:",
        torch.equal(vectors[0, 0], embedding.table.weight[3]),
    )
    print(
        "Repeated 'l' vectors are equal:",
        torch.equal(vectors[0, 2], vectors[0, 3]),
    )
    print("Embedding table is trainable:", embedding.table.weight.requires_grad)

    print("\nTarget shape:", tuple(y.shape))
    print("Target data type:", y.dtype)


if __name__ == "__main__":
    main()