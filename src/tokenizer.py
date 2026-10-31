class CharTokenizer:
    """
    A simple character-level tokenizer.

    It learns a vocabulary from text.
    Each unique character receives one integer ID.
    """

    def __init__(self, text: str):
        if not text:
            raise ValueError("Training text must not be empty.")

        # sorted(...) makes the vocabulary stable and predictable
        unique_chars = sorted(set(text))

        self.stoi = {ch: idx for idx, ch in enumerate(unique_chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """
        Convert text into a list of token IDs.
        """
        ids = []

        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Unknown character: {ch!r}")
            ids.append(self.stoi[ch])

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token IDs back into text.
        """
        chars = []

        for idx in ids:
            if idx not in self.itos:
                raise ValueError(f"Unknown token ID: {idx}")
            chars.append(self.itos[idx])

        return "".join(chars)