import torch

from src.dataset import create_examples
from src.gpt import TinyGPT
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

    with torch.no_grad():
        initial_loss = calculate_loss(model(x), y).item()

    for _ in range(400):
        optimizer.zero_grad()
        loss = calculate_loss(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(x)
        final_loss = calculate_loss(logits, y).item()

    prompt = torch.tensor(
        [tokenizer.encode("h")],
        dtype=torch.long,
    )
    generated = model.generate(prompt, max_new_tokens=10)

    print("Input shape:", tuple(x.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Position table shape:", tuple(model.position_embedding.weight.shape))
    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print("Loss decreased:", final_loss < initial_loss)
    print("Generated shape:", tuple(generated.shape))
    print("Generated text:", repr(tokenizer.decode(generated[0].tolist())))


if __name__ == "__main__":
    main()