from src.tokenizer import CharTokenizer


def main():
    training_text = "hello world"

    tokenizer = CharTokenizer(training_text)

    text = "hello"
    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    print("Training text:", training_text)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Character to ID map:", tokenizer.stoi)
    print("Original text:", text)
    print("Encoded IDs:", token_ids)
    print("Decoded text:", decoded_text)
    print("List length:", len(token_ids))


if __name__ == "__main__":
    main()