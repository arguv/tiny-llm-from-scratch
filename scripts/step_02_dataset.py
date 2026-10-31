from src.dataset import create_examples
from src.tokenizer import CharTokenizer


def main():
    text = "hello world"
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    block_size = 4

    examples = create_examples(token_ids, block_size)

    print("Text:", text)
    print("Token count:", len(token_ids))
    print("Block size:", block_size)
    print("Example count:", len(examples))

    for index, (inputs, targets) in enumerate(examples[:2], start=1):
        print(f"\nExample {index}")
        print("Input IDs:", inputs)
        print("Target IDs:", targets)
        print("Input text:", repr(tokenizer.decode(inputs)))
        print("Target text:", repr(tokenizer.decode(targets)))


if __name__ == "__main__":
    main()