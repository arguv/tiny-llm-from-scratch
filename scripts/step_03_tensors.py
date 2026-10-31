from src.dataset import create_examples
from src.tensors import examples_to_tensors
from src.tokenizer import CharTokenizer


def main():
    tokenizer = CharTokenizer("hello world")
    token_ids = tokenizer.encode("hello world")
    examples = create_examples(token_ids, block_size=4)

    x, y = examples_to_tensors(examples)

    print("Inputs:")
    print(x)

    print("\nTargets:")
    print(y)

    print("\nInput shape:", tuple(x.shape))
    print("Target shape:", tuple(y.shape))
    print("Data type:", x.dtype)
    print("Device:", x.device)

    print("\nFirst input:", x[0].tolist())
    print("First input shape:", tuple(x[0].shape))
    print("First input text:", tokenizer.decode(x[0].tolist()))


if __name__ == "__main__":
    main()