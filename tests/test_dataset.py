import pytest

from src.dataset import create_examples


def test_creates_shifted_examples():
    examples = create_examples([0, 1, 2, 3, 4, 5], block_size=3)

    assert examples == [
        ([0, 1, 2], [1, 2, 3]),
        ([1, 2, 3], [2, 3, 4]),
        ([2, 3, 4], [3, 4, 5]),
    ]


def test_minimum_length_creates_one_example():
    examples = create_examples([0, 1, 2], block_size=2)

    assert examples == [([0, 1], [1, 2])]


def test_rejects_invalid_block_size():
    with pytest.raises(ValueError, match="at least 1"):
        create_examples([0, 1, 2], block_size=0)


def test_rejects_insufficient_tokens():
    with pytest.raises(ValueError, match="block_size \\+ 1"):
        create_examples([0, 1, 2], block_size=3)