import pytest

from src.tokenizer import CharTokenizer


def test_encode_decode_round_trip():
    tokenizer = CharTokenizer("hello world")

    text = "hello"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text
    assert len(ids) == len(text)


def test_vocab_size():
    tokenizer = CharTokenizer("hello world")

    assert tokenizer.vocab_size == 8


def test_unknown_character_fails():
    tokenizer = CharTokenizer("hello world")

    with pytest.raises(ValueError):
        tokenizer.encode("hello!")