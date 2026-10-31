# Tiny LLM from Scratch

A small, incremental project for learning how language models work. The project currently covers character-level tokenization, preparing shifted input/target examples, and converting those examples into PyTorch tensors for next-token prediction.

## Current status

- `CharTokenizer` builds a vocabulary from the characters in a training string.
- Characters are assigned stable IDs in sorted order.
- Text can be encoded to a list of IDs and decoded back to text.
- Unknown characters and token IDs raise `ValueError`.
- `create_examples` creates sliding-window input/target pairs for next-token prediction.
- `examples_to_tensors` converts examples into CPU `torch.long` tensors.
- Tokenizer, dataset, and tensor helpers are covered by pytest tests.

## Project layout

```text
src/tokenizer.py              Character-level tokenizer
scripts/step_01_tokenization.py  Small tokenizer demonstration
src/dataset.py                Next-token training example generation
scripts/step_02_dataset.py   Dataset example demonstration
src/tensors.py                Dataset-to-PyTorch tensor conversion
scripts/step_03_tensors.py   Tensor conversion demonstration
tests/test_tokenizer.py       Automated tests
tests/test_dataset.py         Dataset helper tests
tests/test_tensors.py         Tensor helper tests
requirements.txt              Pinned development dependencies
```

## Setup

The project uses Python 3.10+ and pytest for testing. Create and activate a virtual environment, then install pytest if it is not already available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the demo

From the repository root:

```bash
python scripts/step_01_tokenization.py
```

Example output includes the vocabulary, the character-to-ID mapping, encoded IDs, and the decoded text.

To inspect the next-token training examples:

```bash
python scripts/step_02_dataset.py
```

For a token sequence and a `block_size` of `N`, `create_examples` returns sliding windows of `N` input tokens and the same windows shifted one token to the right as targets.

To convert those examples into PyTorch tensors:

```bash
python scripts/step_03_tensors.py
```

## Run the tests

```bash
python -m pytest
```

## Usage

```python
from src.tokenizer import CharTokenizer

tokenizer = CharTokenizer("hello world")
ids = tokenizer.encode("hello")
text = tokenizer.decode(ids)

assert text == "hello"
print(tokenizer.vocab_size)
```

Dataset example generation can be used after tokenization:

```python
from src.dataset import create_examples

examples = create_examples(ids, block_size=4)
inputs, targets = examples[0]
```

The examples can then be batched into CPU tensors:

```python
from src.tensors import examples_to_tensors

x, y = examples_to_tensors(examples)
```

The training text must not be empty, every character passed to `encode` must appear in the training text, and the token sequence must contain at least `block_size + 1` tokens. `examples_to_tensors` requires at least one example. Local environments, caches, generated datasets, checkpoints, and credentials are excluded by `.gitignore`.
