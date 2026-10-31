def create_examples(
    token_ids: list[int],
    block_size: int,
) -> list[tuple[list[int], list[int]]]:
    """Create input/target pairs for next-token prediction."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    if len(token_ids) <= block_size:
        raise ValueError(
            "Need at least block_size + 1 tokens to create an example."
        )

    examples = []

    for start in range(len(token_ids) - block_size):
        inputs = token_ids[start : start + block_size]
        targets = token_ids[start + 1 : start + block_size + 1]
        examples.append((inputs, targets))

    return examples