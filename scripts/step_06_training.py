import torch

from src.dataset import create_examples
from src.model import BigramLanguageModel
from src.tensors import examples_to_tensors
from src.tokenizer import CharTokenizer
from src.training import train_model


def main():
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

    initial_loss, final_loss = train_model(
        model,
        x,
        y,
        steps=200,
        learning_rate=0.05,
    )

    print("Input shape:", tuple(x.shape))
    print("Target shape:", tuple(y.shape))
    print("Training updates:", 200)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print("Loss decreased:", final_loss < initial_loss)

    model.eval()

    with torch.no_grad():
        h = torch.tensor(
            [tokenizer.encode("h")],
            dtype=torch.long,
        )
        logits = model(h)
        predicted_id = logits[0, 0].argmax().item()

    print("Predicted next character after 'h':", tokenizer.decode([predicted_id]))


if __name__ == "__main__":
    main()