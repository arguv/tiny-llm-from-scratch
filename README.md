# Tiny LLM from Scratch

A small, incremental project for learning how language models work. The project currently covers character-level tokenization, preparing shifted input/target examples, converting those examples into PyTorch tensors, looking up trainable token embeddings, producing next-token scores with a tiny bigram language model, training that model with cross-entropy loss, generating text one token at a time, and implementing causal self-attention.

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
- `BigramLanguageModel` maps each token embedding to a score for every vocabulary item.
- `calculate_loss` measures next-token prediction error with cross-entropy.
- `train_model` updates model parameters with Adam and reports the initial and final loss.
- `generate` repeatedly selects the highest-scoring next token and appends it to a prompt.
- `CausalSelfAttention` uses scaled dot-product attention while blocking future positions.
- Tokenizer, dataset, tensor, embedding, model, training, generation, and attention components are covered by pytest tests.

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
src/model.py                  Tiny bigram language model
scripts/step_05_model.py      Model forward-pass demonstration
src/training.py               Loss calculation and training loop
scripts/step_06_training.py   Model training demonstration
src/generation.py             Greedy autoregressive generation
scripts/step_07_generation.py Text generation demonstration
src/attention.py              Causal self-attention component
scripts/step_08_attention.py  Attention masking demonstration
tests/test_tokenizer.py       Automated tests
tests/test_dataset.py         Dataset helper tests
tests/test_tensors.py         Tensor helper tests
tests/test_embeddings.py      Embedding component tests
tests/test_model.py            Model tests
tests/test_training.py        Training helper tests
tests/test_generation.py      Generation helper tests
tests/test_attention.py       Attention component tests
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

To run the first next-token model:

```bash
python scripts/step_05_model.py
```

The model embeds each input token and passes the resulting vectors through a linear layer. For an input tensor shaped `(batch, sequence)`, it returns logits shaped `(batch, sequence, vocabulary)`: one score for every possible next token at every position. Taking `argmax(dim=-1)` selects the highest-scoring token ID, but the model is untrained, so these predictions are not meaningful yet.

To train the bigram model on the example data:

```bash
python scripts/step_06_training.py
```

The training demo runs 200 Adam updates, prints the initial and final cross-entropy loss, and checks that the loss decreased. It then switches the model to evaluation mode and predicts the next character after `h`.

To generate text from the trained model:

```bash
python scripts/step_07_generation.py
```

The generation demo trains the model, starts with the prompt `h`, and appends 10 predicted characters. Generation uses greedy decoding: at each step it selects the highest-scoring token, runs without gradients, preserves the original prompt, and leaves model parameters unchanged.

To inspect causal self-attention:

```bash
python scripts/step_08_attention.py
```

The attention demo projects token embeddings into queries, keys, and values, computes scaled dot-product scores, masks future positions, and prints the resulting attention weights. Each row of weights sums to one, and no token can attend to a later token.

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

The embedding output can be passed through the tiny bigram model:

```python
from src.model import BigramLanguageModel

model = BigramLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=6,
)
logits = model(x)
predicted_ids = logits.argmax(dim=-1)

assert tuple(logits.shape) == (len(examples), 4, tokenizer.vocab_size)
assert tuple(predicted_ids.shape) == (len(examples), 4)
```

This first model uses only the current token’s embedding when producing its scores. It has no positional information or context window beyond the current token. The causal attention component exists as a separate building block and is not integrated into `BigramLanguageModel` yet.

The model can be trained with the helpers in `src/training.py`:

```python
from src.training import calculate_loss, train_model

initial_loss, final_loss = train_model(
    model,
    x,
    y,
    steps=200,
    learning_rate=0.05,
)

assert final_loss < initial_loss
```

`calculate_loss` flattens the model’s `(batch, sequence, vocabulary)` logits and the `(batch, sequence)` target IDs before applying cross-entropy. `train_model` performs the forward pass, loss calculation, backpropagation, and Adam update for each step. It rejects `steps < 1` and returns the loss before and after training.

After training, text can be generated from a nonempty batch of prompt IDs:

```python
import torch

from src.generation import generate

prompt_ids = torch.tensor([tokenizer.encode("h")], dtype=torch.long)
generated_ids = generate(model, prompt_ids, max_new_tokens=10)
generated_text = tokenizer.decode(generated_ids[0].tolist())
```

`generate` clones the prompt, repeatedly evaluates only the most recent token, appends the selected token, and returns the complete prompt-plus-continuation tensor. It rejects empty prompts and negative token counts. Because this is a bigram model, each new prediction depends only on the immediately preceding token.

Attention can be applied to a sequence of embedding vectors:

```python
from src.attention import CausalSelfAttention

attention = CausalSelfAttention(embedding_dim=6)
attention_output, weights = attention(vectors)

assert tuple(attention_output.shape) == tuple(vectors.shape)
assert tuple(weights.shape) == (len(examples), 4, 4)
```

For input shape `(batch, sequence, embedding)`, the output has the same shape and the attention weights have shape `(batch, sequence, sequence)`. The causal lower-triangular mask ensures position `i` can use positions `0` through `i`, but never a future position.

The training text must not be empty, every character passed to `encode` must appear in the training text, and the token sequence must contain at least `block_size + 1` tokens. `examples_to_tensors` requires at least one example, `generate` requires a nonempty two-dimensional prompt tensor, and attention requires a three-dimensional sequence tensor. Local environments, caches, generated datasets, checkpoints, and credentials are excluded by `.gitignore`.
