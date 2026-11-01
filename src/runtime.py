import torch

from src.gpt import TinyGPT


@torch.no_grad()
def inspect_runtime(
    model: TinyGPT,
    token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if token_ids.ndim != 2 or token_ids.shape[1] == 0:
        raise ValueError("Provide a nonempty sequence with two dimensions.")

    model.eval()

    context = token_ids[:, -model.block_size:]
    positions = torch.arange(
        context.shape[1],
        device=context.device,
    )

    token_vectors = model.token_embedding(context)
    position_vectors = model.position_embedding(positions)
    combined = token_vectors + position_vectors

    block_output = model.block(combined)
    normalized = model.final_norm(block_output)
    logits = model.output(normalized)

    last_logits = logits[:, -1, :]
    probabilities = torch.softmax(last_logits, dim=-1)
    next_token = probabilities.argmax(dim=-1, keepdim=True)

    return {
        "context": context,
        "positions": positions,
        "token_vectors": token_vectors,
        "position_vectors": position_vectors,
        "combined": combined,
        "block_output": block_output,
        "normalized": normalized,
        "logits": logits,
        "last_logits": last_logits,
        "probabilities": probabilities,
        "next_token": next_token,
    }