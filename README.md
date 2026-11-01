# Tiny LLM from Scratch

A small, incremental project for learning how language models work. The project currently covers character-level tokenization, preparing shifted input/target examples, converting those examples into PyTorch tensors, and looking up trainable token embeddings for next-token prediction.

## Important distinctions

- Tokenization is text preprocessing, usually outside the neural network.
- Embeddings are learned numerical representations.
- Attention is a mechanism inside the Transformer.
- Transformer is the neural network architecture combining these mechanisms.
- LLM is the complete trained language-model system, including its learned parameters and inference process.

## Current status

- `CharTokenizer` builds a vocabulary from the characters in a training string.
- Characters are assigned stable IDs in sorted order.
- Text can be encoded to a list of IDs and decoded back to text.
- Unknown characters and token IDs raise `ValueError`.
- `create_examples` creates sliding-window input/target pairs for next-token prediction.
- `examples_to_tensors` converts examples into CPU `torch.long` tensors.
- `TokenEmbedding` maps token IDs to trainable CPU floating-point vectors.
- Tokenizer, dataset, tensor, and embedding helpers are covered by pytest tests.

## Project layout

```text
src/tokenizer.py              Character-level tokenizer
scripts/step_01_tokenization.py  Small tokenizer demonstration
src/dataset.py                Next-token training example generation
scripts/step_02_dataset.py   Dataset example demonstration
src/tensors.py                Dataset-to-PyTorch tensor conversion
scripts/step_03_tensors.py   Tensor conversion demonstration
src/embeddings.py             Trainable token embedding component
scripts/step_04_embeddings.py Embedding lookup demonstration
tests/test_tokenizer.py       Automated tests
tests/test_dataset.py         Dataset helper tests
tests/test_tensors.py         Tensor helper tests
tests/test_embeddings.py      Embedding component tests
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

To look up a six-value trainable vector for every input token:

```bash
python scripts/step_04_embeddings.py
```

The embedding demo reports the `(8, 6)` embedding table, the `(7, 4, 6)` output for the example batch, and verifies that repeated token IDs retrieve the same table row. Embeddings are initialized randomly and are not trained yet.

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

The input tensor can be passed through a token embedding component:

```python
from src.embeddings import TokenEmbedding

embedding = TokenEmbedding(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=6,
)
vectors = embedding(x)

assert tuple(vectors.shape) == (len(examples), 4, 6)
```

The embedding layer preserves the batch and sequence dimensions and adds one dimension for each token vector. It does not yet make predictions or train the parameters.

The training text must not be empty, every character passed to `encode` must appear in the training text, and the token sequence must contain at least `block_size + 1` tokens. `examples_to_tensors` requires at least one example. Local environments, caches, generated datasets, checkpoints, and credentials are excluded by `.gitignore`.
