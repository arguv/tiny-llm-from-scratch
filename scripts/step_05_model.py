import torch

from src.dataset import create_examples
from src.model import BigramLanguageModel
from src.tensors import examples_to_tensors
from src.tokenizer import CharTokenizer


def main():
    torch.manual_seed(42)

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

    logits = model(x)
    predicted_ids = logits.argmax(dim=-1)

    print("Input shape:", tuple(x.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Predicted IDs shape:", tuple(predicted_ids.shape))
    print("Target shape:", tuple(y.shape))

    print("\nScores per position:", logits.shape[-1])
    print("First position scores shape:", tuple(logits[0, 0].shape))
    print("First example prediction count:", predicted_ids[0].numel())

    print("\nFirst input:", tokenizer.decode(x[0].tolist()))
    print("First target:", tokenizer.decode(y[0].tolist()))
    print("Trainable parameter count:", sum(
        parameter.numel() for parameter in model.parameters()
    ))

    print(
        "Repeated 'l' positions have equal scores:",
        torch.equal(logits[0, 2], logits[0, 3]),
    )


if __name__ == "__main__":
    main()