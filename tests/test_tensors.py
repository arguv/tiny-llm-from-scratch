import pytest
import torch

from src.dataset import create_examples
from src.tensors import examples_to_tensors
from src.tokenizer import CharTokenizer


def test_conversion_preserves_values_and_shape():
    tokenizer = CharTokenizer("hello world")
    examples = create_examples(tokenizer.encode("hello world"), block_size=4)

    x, y = examples_to_tensors(examples)

    assert tuple(x.shape) == (7, 4)
    assert tuple(y.shape) == (7, 4)

    assert x.tolist() == [inputs for inputs, targets in examples]
    assert y.tolist() == [targets for inputs, targets in examples]

    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"


def test_empty_examples_fail():
    with pytest.raises(ValueError, match="must not be empty"):
        examples_to_tensors([])