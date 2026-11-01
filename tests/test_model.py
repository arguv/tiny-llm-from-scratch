import torch

from src.model import BigramLanguageModel


def test_model_returns_scores_for_every_token():
    model = BigramLanguageModel(vocab_size=8, embedding_dim=6)
    token_ids = torch.tensor(
        [[3, 2, 4, 4], [2, 4, 4, 5]],
        dtype=torch.long,
    )

    logits = model(token_ids)

    assert tuple(logits.shape) == (2, 4, 8)
    assert torch.isfinite(logits).all().item()
    assert logits.device.type == "cpu"

    predicted_ids = logits.argmax(dim=-1)

    assert tuple(predicted_ids.shape) == (2, 4)
    assert ((predicted_ids >= 0) & (predicted_ids < 8)).all().item()


def test_prediction_depends_only_on_current_token():
    model = BigramLanguageModel(vocab_size=8, embedding_dim=6)
    token_ids = torch.tensor(
        [[3, 2, 4], [7, 5, 4]],
        dtype=torch.long,
    )

    logits = model(token_ids)

    assert torch.equal(logits[0, 2], logits[1, 2])