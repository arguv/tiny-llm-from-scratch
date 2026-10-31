import torch


def examples_to_tensors(
    examples: list[tuple[list[int], list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert equally sized input/target examples into CPU tensors."""
    if not examples:
        raise ValueError("examples must not be empty.")

    inputs = [input_ids for input_ids, target_ids in examples]
    targets = [target_ids for input_ids, target_ids in examples]

    x = torch.tensor(inputs, dtype=torch.long, device="cpu")
    y = torch.tensor(targets, dtype=torch.long, device="cpu")

    return x, y