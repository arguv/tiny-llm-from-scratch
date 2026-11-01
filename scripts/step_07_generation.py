import torch

from src.dataset import create_examples
from src.generation import generate
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
    train_model(model, x, y, steps=200, learning_rate=0.05)

    prompt = "h"
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=10,
    )
    text = tokenizer.decode(generated_ids[0].tolist())

    parameters_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )

    print("Prompt:", prompt)
    print("Prompt shape:", tuple(prompt_ids.shape))
    print("Generated shape:", tuple(generated_ids.shape))
    print("New characters:", generated_ids.shape[1] - prompt_ids.shape[1])
    print("Prompt preserved:", text.startswith(prompt))
    print("Parameters unchanged:", parameters_unchanged)
    print("Generated text:", repr(text))


if __name__ == "__main__":
    main()