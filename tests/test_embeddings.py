import torch

from src.embeddings import TokenEmbedding


def test_embedding_looks_up_table_rows():
    embedding = TokenEmbedding(vocab_size=8, embedding_dim=6)
    token_ids = torch.tensor([[3, 2, 4, 4]], dtype=torch.long)

    vectors = embedding(token_ids)

    assert tuple(embedding.table.weight.shape) == (8, 6)
    assert tuple(vectors.shape) == (1, 4, 6)

    for position, token_id in enumerate([3, 2, 4, 4]):
        assert torch.equal(
            vectors[0, position],
            embedding.table.weight[token_id],
        )

    assert vectors.dtype == torch.float32
    assert vectors.device.type == "cpu"


def test_repeated_ids_share_the_same_vector():
    embedding = TokenEmbedding(vocab_size=8, embedding_dim=6)
    token_ids = torch.tensor([[4, 4], [4, 4]], dtype=torch.long)

    vectors = embedding(token_ids)

    assert torch.equal(vectors[0, 0], vectors[0, 1])
    assert torch.equal(vectors[0, 0], vectors[1, 0])
    assert embedding.table.weight.requires_grad