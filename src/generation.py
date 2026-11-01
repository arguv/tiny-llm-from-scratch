import torch

from src.model import BigramLanguageModel


@torch.no_grad()
def generate(
    model: BigramLanguageModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    if token_ids.ndim != 2 or token_ids.shape[1] == 0:
        raise ValueError("token_ids must have shape (batch, nonempty sequence).")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative.")

    model.eval()
    generated = token_ids.clone()

    for _ in range(max_new_tokens):
        last_token = generated[:, -1:]
        logits = model(last_token)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

    return generated