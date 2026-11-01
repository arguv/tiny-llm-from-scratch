import pytest
import torch

from src.dataset import create_examples
from src.model import BigramLanguageModel
from src.tensors import examples_to_tensors
from src.tokenizer import CharTokenizer
from src.training import train_model


def test_training_reduces_loss_and_updates_embeddings():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    tokenizer = CharTokenizer("hello world")
    examples = create_examples(
        tokenizer.encode("hello world"),
        block_size=4,
    )
    x, y = examples_to_tensors(examples)

    model = BigramLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=6,
    )
    original_embeddings = model.embedding.table.weight.detach().clone()

    initial_loss, final_loss = train_model(
        model,
        x,
        y,
        steps=200,
        learning_rate=0.05,
    )

    assert final_loss < initial_loss * 0.7
    assert not torch.equal(
        original_embeddings,
        model.embedding.table.weight.detach(),
    )


def test_training_rejects_zero_steps():
    model = BigramLanguageModel(vocab_size=8, embedding_dim=6)
    x = torch.tensor([[3]], dtype=torch.long)
    y = torch.tensor([[2]], dtype=torch.long)

    with pytest.raises(ValueError, match="at least 1"):
        train_model(model, x, y, steps=0)