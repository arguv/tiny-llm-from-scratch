# Tiny LLM from Scratch

A small, incremental project for learning how language models work. The first step is a deterministic character-level tokenizer that converts text to integer IDs and back again.

## Current status

- `CharTokenizer` builds a vocabulary from the characters in a training string.
- Characters are assigned stable IDs in sorted order.
- Text can be encoded to a list of IDs and decoded back to text.
- Unknown characters and token IDs raise `ValueError`.
- The tokenizer is covered by pytest tests.

## Project layout

```text
src/tokenizer.py              Character-level tokenizer
scripts/step_01_tokenization.py  Small tokenizer demonstration
tests/test_tokenizer.py       Automated tests
```

## Setup

The project uses Python 3.10+ and pytest for testing. Create and activate a virtual environment, then install pytest if it is not already available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest
```

## Run the demo

From the repository root:

```bash
python scripts/step_01_tokenization.py
```

Example output includes the vocabulary, the character-to-ID mapping, encoded IDs, and the decoded text.

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

The training text must not be empty, and every character passed to `encode` must appear in the training text. Large datasets, model checkpoints, and credentials should remain outside version control.
