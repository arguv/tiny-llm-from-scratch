import torch

from src.dataset import create_examples
from src.gpt import TinyGPT
from src.runtime import inspect_runtime
from src.tensors import examples_to_tensors
from src.tokenizer import CharTokenizer
from src.training import calculate_loss


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    tokenizer = CharTokenizer("hello world")
    examples = create_examples(
        tokenizer.encode("hello world"),
        block_size=4,
    )
    x, y = examples_to_tensors(examples)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=6,
        block_size=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for _ in range(400):
        optimizer.zero_grad()
        loss = calculate_loss(model(x), y)
        loss.backward()
        optimizer.step()

    prompt = "hell"
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
    )
    trace = inspect_runtime(model, prompt_ids)

    print("Prompt:", repr(prompt))
    print("Context IDs:", trace["context"].tolist())

    print("\nRuntime tensor shapes:")
    for name, tensor in trace.items():
        print(f"{name}: {tuple(tensor.shape)}")

    print(
        "\nProbabilities sum to one:",
        torch.allclose(
            trace["probabilities"].sum(dim=-1),
            torch.ones(1),
        ),
    )

    top_probabilities, top_ids = trace["probabilities"][0].topk(3)

    print("\nTop next-character candidates:")
    for probability, token_id in zip(
        top_probabilities.tolist(),
        top_ids.tolist(),
    ):
        character = tokenizer.decode([token_id])
        print(f"{character!r}: {probability:.4f}")

    extended = torch.cat(
        (prompt_ids, trace["next_token"]),
        dim=1,
    )
    print(
        "\nExtended text:",
        repr(tokenizer.decode(extended[0].tolist())),
    )
    print("Extended shape:", tuple(extended.shape))


if __name__ == "__main__":
    main()