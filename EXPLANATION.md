# Step 1 — Tokenization basics

## Goal

Step 1’s goal was to convert text into numbers and then convert those numbers back into the original text. This prepares text for a language model, which performs calculations on numbers.

We built a **character-level tokenizer**: each character—including spaces—is one token.

## 1. Build the vocabulary

We gave the tokenizer the text `"hello world"`.

The expression `sorted(set(text))` does two things:

- `set(text)` removes repeated characters.
- `sorted(...)` puts the remaining characters in a predictable order.

The result contains **8 unique characters**:

| Character | Assigned ID |
|---|---:|
| space | 0 |
| `d` | 1 |
| `e` | 2 |
| `h` | 3 |
| `l` | 4 |
| `o` | 5 |
| `r` | 6 |
| `w` | 7 |

This is our **vocabulary**. Its size is **8**, although `"hello world"` contains 11 characters.

## 2. Create two lookup dictionaries

Inside `CharTokenizer`, we created:

| Dictionary | Purpose | Example |
|---|---|---|
| `stoi` — string to integer | Find the ID for a character | `"h"` → `3` |
| `itos` — integer to string | Find the character for an ID | `3` → `"h"` |

These dictionaries let us move in both directions.

**The IDs are labels.** ID `5` does not mean that `"o"` has a larger value or more meaning than `"h"` with ID `3`.

## 3. Encode text into IDs

Calling `tokenizer.encode("hello")` looks up each character:

| Character | `h` | `e` | `l` | `l` | `o` |
|---|---:|---:|---:|---:|---:|
| ID | 3 | 2 | 4 | 4 | 5 |

The result is **`[3, 2, 4, 4, 5]`**.

Repeated characters receive the same ID. Both occurrences of `"l"` become `4`.

If a character is absent from the vocabulary, such as `"!"`, our tokenizer raises `ValueError`.

## 4. Decode IDs back into text

Calling `tokenizer.decode([3, 2, 4, 4, 5])` uses the reverse dictionary:

**`[3, 2, 4, 4, 5]` → `"hello"`**

This confirms that our conversion preserves the original text.

## 5. Verify the goal with tests

Our three tests checked:

| Test | What it proves |
|---|---|
| Encode, then decode | We recover the original text |
| Vocabulary size | `"hello world"` contains 8 unique characters |
| Unknown character | Unsupported input raises an error |

## What we achieved

A working, reversible conversion between text and token IDs.

We have not trained a model yet. The tokenizer only assigns and looks up IDs; it does not understand words or predict characters. These IDs are the data we will use in the following steps.

---

# Step 2 — Prepare input and target sequences

## Goal

Step 2’s goal was to turn token IDs into training examples so a model can learn to predict the next character.

In Step 1, we converted text into numbers. In Step 2, we organized those numbers into **inputs and expected answers**, called **targets**.

## 1. Start with token IDs

We used the same tokenizer to encode `"hello world"`:

| Character | `h` | `e` | `l` | `l` | `o` | space | `w` | `o` | `r` | `l` | `d` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Token ID | 3 | 2 | 4 | 4 | 5 | 0 | 7 | 5 | 6 | 4 | 1 |

We now have **11 token IDs**. Their order matches the original text.

## 2. Choose a block size

We set **`block_size = 4`**.

This means each training example contains:

- An **input list** of 4 token IDs.
- A **target list** of 4 token IDs.

To create both lists, we need **5 consecutive tokens**: the target extends one position beyond the input.

## 3. Create the first input and target

We took the first four characters as input and shifted forward one position to get the targets:

| Sequence | Text | Token IDs |
|---|---|---|
| Input | `"hell"` | `[3, 2, 4, 4]` |
| Target | `"ello"` | `[2, 4, 4, 5]` |

**Each target is the character immediately after its corresponding input character.**

For a future model that can use preceding context, the example represents these predictions:

| Available input context | Expected next character |
|---|---|
| `"h"` | `"e"` |
| `"he"` | `"l"` |
| `"hel"` | `"l"` |
| `"hell"` | `"o"` |

The dataset function only prepares the lists. Later, we’ll build the model so each prediction can use preceding characters without seeing future ones.

## 4. Slide forward to create more examples

Our `create_examples()` function moves the starting position forward by one token each time:

| Example | Input text | Target text |
|---|---|---|
| 1 | `"hell"` | `"ello"` |
| 2 | `"ello"` | `"llo "` |
| 3 | `"llo "` | `"lo w"` |
| 4 | `"lo w"` | `"o wo"` |
| 5 | `"o wo"` | `" wor"` |
| 6 | `" wor"` | `"worl"` |
| 7 | `"worl"` | `"orld"` |

Spaces count as characters, so they appear in our examples too.

We created **7 examples**: **11 tokens − block size 4 = 7**.

We cannot create another complete example after the seventh because its target would extend beyond the text.

## 5. Understand how the code extracts the lists

Inside the loop, we used Python slicing:

| Expression | What it extracts |
|---|---|
| `token_ids[start : start + block_size]` | The input block |
| `token_ids[start + 1 : start + block_size + 1]` | The target block, shifted one position forward |

Python slices **include the starting index and exclude the ending index**.

For the first example, `start = 0`:

- Input uses indices **0, 1, 2, 3**.
- Target uses indices **1, 2, 3, 4**.

The function stores each pair as `(inputs, targets)` and returns a list containing all pairs.

## 6. Verify the goal with tests

We added four tests:

| Test | What it proves |
|---|---|
| Exact shifted examples | Targets are shifted one position forward, including in the final example |
| Minimum valid length | `block_size + 1` tokens produce exactly one example |
| Invalid block size | A block size of zero raises `ValueError` |
| Insufficient tokens | Too few tokens raise `ValueError` |

Together with Step 1’s three tests, the expected total is **7 passed**.

## What we achieved

A dataset of input sequences paired with the correct next-token answers.

We still have **no tensors and no trained model**. We have prepared the examples a model will learn from: **“Given these characters, predict what comes next at each position.”**

---

# Step 3 — Convert the dataset into PyTorch tensors

## Goal

Convert Step 2’s Python lists into **PyTorch tensors**, the numeric containers our model will use. The IDs and their order stay the same; we change their container.

## 1. Start with the examples

Step 2 produced 7 pairs, each containing 4 input IDs and 4 target IDs. The first pair is:

| Part | IDs | Text |
|---|---|---|
| Input | `[3, 2, 4, 4]` | `"hell"` |
| Target | `[2, 4, 4, 5]` | `"ello"` |

## 2. Separate the inputs and targets

In `src/tensors.py`, `examples_to_tensors()` first rejects an empty dataset.

Then it collects the two sides separately:

| Expression | Result |
|---|---|
| `[input_ids for input_ids, target_ids in examples]` | All input lists, arranged as rows |
| `[target_ids for input_ids, target_ids in examples]` | All target lists, arranged as rows |

Each iteration unpacks a pair into `input_ids` and `target_ids`. Each resulting nested list contains 7 rows of 4 integers.

## 3. Create the tensors

`torch.tensor(inputs, dtype=torch.long, device="cpu")` creates the input tensor `x`. The equivalent expression using `targets` creates `y`.

| Code element | Purpose |
|---|---|
| `torch.tensor(...)` | Create a PyTorch tensor from the numbers |
| `dtype=torch.long` | Store IDs as 64-bit integers, printed as `torch.int64` |
| `device="cpu"` | Keep the tensor in CPU memory |
| `x` | Input tensor |
| `y` | Correct-target tensor |

The function returns both with `return x, y`. In the script, `x, y = examples_to_tensors(examples)` receives them.

The IDs remain labels. We have not converted them into embedding vectors yet.

## 4. Understand the shapes

Both tensors have shape **`(7, 4)`**:

| Dimension | Size | Meaning |
|---|---:|---|
| First | 7 | Number of examples |
| Second | 4 | Number of token positions per example |

Each tensor contains 28 IDs. Shape describes how values are arranged; it does not describe vocabulary size.

## 5. Inspect a row

In `scripts/step_03_tensors.py`, we inspect the results:

| Expression | Result | Meaning |
|---|---|---|
| `tuple(x.shape)` | `(7, 4)` | Seven rows, four columns |
| `x[0]` | `tensor([3, 2, 4, 4])` | First input row |
| `tuple(x[0].shape)` | `(4,)` | One-dimensional sequence of four IDs |
| `x[0].tolist()` | `[3, 2, 4, 4]` | Convert the row back into a Python list |
| `tokenizer.decode(x[0].tolist())` | `"hell"` | Recover readable text with our tokenizer |

Selecting one row removes the row dimension: `(7, 4)` becomes `(4,)`.

## 6. Why tensors?

PyTorch model layers operate on tensors. Tensors support numerical operations on collections of values and integrate with automatic differentiation, which we will use for training.

Our integer token tensors do not require gradients. Later, the model’s trainable floating-point parameters will.

This step only converts and inspects data. It makes no predictions and trains no model.

## 7. Installation and NumPy

Step 3 introduced PyTorch’s CPU build inside the existing project `.venv`.

The missing-NumPy warning occurred when PyTorch tried to initialize its NumPy integration. NumPy is a library for numerical arrays. Our code does not directly use it, so the script still completed successfully. Installing NumPy in the same environment makes that integration available and removes the missing-NumPy warning.

Activate `.venv` each time you open a new terminal. Packages remain installed when the terminal closes; activation selects that environment for the current terminal session.

## 8. Verify with tests

| Test | What it proves |
|---|---|
| Conversion preserves values and shape | All input and target values are preserved; tensors have shape `(7, 4)`, integer type, and CPU location |
| Empty examples fail | An empty dataset raises `ValueError` before creating tensors |

Together with the previous seven tests, the expected total is **9 passed**.

## What we achieved

**Text → token IDs → input/target pairs → input and target tensors**

Our dataset is now in the format PyTorch model operations will consume. The tensors exist in memory while the program runs; we have not saved them to disk. No model has been trained yet.

---

# Step 4 — Embeddings: turn token IDs into vectors

## Goal

Convert every input token ID into a trainable vector of floating-point numbers.

In Step 3, our inputs were integer tensors. In Step 4, an embedding layer looks up a vector for each integer ID. No model training happens in this step.

## 1. Where this fits in our project

We are building a tiny GPT-style language model from scratch using Python and PyTorch on CPU. It demonstrates the architecture behind LLMs, but is much smaller than models typically called SLMs (Small Language Models).

| Stage | What we build |
|---|---|
| Tokenization | Convert text to IDs and back |
| Dataset preparation | Pair input sequences with correct next-token targets |
| Tensors | Store the sequences in PyTorch numeric containers |
| Embeddings — current stage | Give each token ID a trainable vector |
| First model and training | Produce predictions and adjust parameters using errors |
| Generation | Predict and append tokens repeatedly |
| Attention and transformers | Process relationships between tokens in context |
| Runtime execution | Follow what happens during generation |

**An embedding layer is one component inside the language model.** It is not a complete language model.

## 2. Why use vectors?

Token IDs are labels. ID 5 is not inherently more meaningful than ID 3, and their numerical distance does not describe character similarity.

An embedding gives each ID a vector whose values training can adjust. These vectors become the numeric representations that later model layers process.

The initial values are random; they have not learned meaningful relationships.

## 3. Create an embedding table

Our vocabulary has 8 entries. We choose 6 numbers per token, so the table has shape **(8, 6)**.

| Setting | Value | Meaning |
|---|---:|---|
| vocab_size | 8 | Number of rows: one per token ID |
| embedding_dim | 6 | Number of floating-point values per row |

Six is a small demonstration choice, not a requirement. Changing it changes the vector length.

## 4. What the code creates

In `src/embeddings.py`, `TokenEmbedding` inherits from `nn.Module`, PyTorch’s base class for model components. It tracks trainable parameters.

| Code | Purpose |
|---|---|
| `super().__init__()` | Initialize the parent module |
| `nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device="cpu")` | Create the trainable lookup table on CPU |
| `self.table` | Store the embedding layer as part of our component |
| `self.table.weight` | Access its matrix of trainable values |
| `forward(token_ids)` | Describe how inputs pass through the component |
| `return self.table(token_ids)` | Look up the vector for every input ID |

Calling `embedding(x)` invokes PyTorch’s module machinery, which runs our `forward()` method.

## 5. How lookup works

For the first input, `[3, 2, 4, 4]`, representing "hell":

| Position | Character | ID | Retrieved vector |
|---|---|---:|---|
| 0 | h | 3 | Row 3 |
| 1 | e | 2 | Row 2 |
| 2 | l | 4 | Row 4 |
| 3 | l | 4 | Row 4 again |

The ID selects a row; it is not multiplied by the row.

Both occurrences of l get identical vectors. At this stage the layer considers token identity only, without position or surrounding context.

## 6. Understand the output shape

In `scripts/step_04_embeddings.py`, we reuse our existing pipeline to obtain `x` and `y`. Then `vectors = embedding(x)` performs the lookup.

| Tensor | Shape | Meaning |
|---|---|---|
| Embedding table | (8, 6) | 8 token vectors, each containing 6 values |
| Input x | (7, 4) | 7 examples with 4 token IDs each |
| Output vectors | (7, 4, 6) | A 6-value vector for each input position |
| vectors[0] | (4, 6) | Vectors for the first example |
| vectors[0, 0] | (6,) | Vector for its first token |
| Target y | (7, 4) | Correct next-token IDs |

The lookup preserves the input dimensions and adds the vector dimension.

Inputs and targets remain integer tensors. The embedding output uses floating-point numbers, normally `torch.float32` in our script.

Targets stay as IDs because later they will identify the correct next-token classes when we calculate prediction errors.

## 7. Trainable does not mean trained

`embedding.table.weight.requires_grad` is True: PyTorch can calculate gradients for these parameters during training.

We have not calculated a loss, run backpropagation, or updated any parameters. The table remains at its initial values.

`torch.manual_seed(42)` makes random initialization repeatable in the same environment. It does not teach the embeddings anything.

## 8. Verify the lookup

The script checks:

- The first output vector equals table row 3.
- The repeated l tokens retrieve equal vectors.
- The table is trainable.
- Tensor shapes and CPU location match expectations.

In `tests/test_embeddings.py`, two tests check the lookup against the actual table rows and verify repeated IDs share a vector.

| Test case | Input shape | Output shape |
|---|---|---|
| Row lookup | (1, 4) | (1, 4, 6) |
| Repeated IDs | (2, 2) | (2, 2, 6) |

With all previous tests present and passing, the expected total is **11 passed**.

## 9. Installations and storage

No new installations are required for Step 4. Activate the existing project .venv and use the PyTorch installed earlier.

The table and output tensors live in memory during execution. We have not saved model parameters to disk. Running the script creates a new table; the seed makes its initialization repeatable in the same environment.

## What we achieved

**Text → token IDs → input/target pairs → tensors → token vectors**

Every input token now has a floating-point representation that future training can adjust. We have not yet built a complete next-token predictor, trained the table, or generated text.


---

# Step 5 — Build the first next-token model

## Goal

Turn Step 4’s embedding vectors into **scores for the next token**.

We now have a complete forward computation that accepts token IDs and produces next-token scores. We have not trained the model, so its initial predictions are arbitrary.

## 1. What was missing after Step 4?

An embedding represents an input token as a vector of 6 floating-point numbers. That vector does not directly tell us which of our 8 vocabulary tokens should come next.

We add an output layer that converts each 6-number vector into 8 scores, one for each possible next token.

| Stage | Shape | Meaning |
|---|---|---|
| Input IDs | (7, 4) | Seven examples, four tokens each |
| Embedding vectors | (7, 4, 6) | Six values for every input token |
| Output scores | (7, 4, 8) | Eight next-token scores at every position |

## 2. What the code builds

In `src/model.py`, `BigramLanguageModel` inherits from `nn.Module` and contains two components:

| Component | Code | Purpose |
|---|---|---|
| Token representation | `TokenEmbedding(vocab_size, embedding_dim)` | Look up a vector for each input ID |
| Output layer | `nn.Linear(in_features=embedding_dim, out_features=vocab_size, device="cpu")` | Convert each vector into vocabulary scores |

In `forward()`, `vectors = self.embedding(token_ids)` performs the lookup. Then `logits = self.output(vectors)` calculates the scores. Finally, `return logits` sends them to the caller.

Calling `model(x)` runs this computation through PyTorch’s module machinery.

## 3. How the linear layer calculates scores

Each of the 8 output scores has 6 trainable weights and one trainable bias.

For one score, the layer multiplies each of the 6 embedding values by its corresponding weight, adds the products, and adds the bias.

It repeats this calculation for every vocabulary entry. The same output-layer parameters are used at all token positions.

The layer changes only the last dimension from 6 to 8. It does not combine information from neighboring positions.

## 4. What are logits?

**Logits are raw scores**, not probabilities. They can be positive or negative and do not need to sum to one.

| Score index | Candidate next character |
|---|---|
| 0 | space |
| 1 | d |
| 2 | e |
| 3 | h |
| 4 | l |
| 5 | o |
| 6 | r |
| 7 | w |

A higher score means the model currently favors that candidate more. Later, we can turn logits into probabilities.

## 5. Select the highest-scoring token

In `scripts/step_05_model.py`:

| Expression | Purpose | Result shape |
|---|---|---|
| `logits = model(x)` | Calculate next-token scores | (7, 4, 8) |
| `predicted_ids = logits.argmax(dim=-1)` | Select the index of the highest score at each position | (7, 4) |
| `logits[0, 0]` | Inspect scores for the first position of the first example | (8,) |
| `predicted_ids[0]` | Inspect four predicted IDs for the first example | (4,) |

`dim=-1` means the last dimension: the 8 candidate scores. Selecting one index from those scores removes that dimension.

If index 2 has the highest score, the predicted next-token ID is 2, which our tokenizer maps to e.

The script processes existing input positions. It does not yet append tokens repeatedly to generate new text.

## 6. Why is this a bigram model?

The model predicts the next token using **only the current token**. It models adjacent token pairs.

For the input "hell", the two l positions have the same ID, retrieve the same embedding, and pass through the same output layer. They therefore produce identical scores.

But the correct next character differs: the first l is followed by l, and the second is followed by o. This model cannot distinguish those contexts.

A block size of 4 lets us process four positions together. It does not make this model use all four tokens as context for each prediction. Attention will later allow information from preceding positions to contribute.

## 7. Count the trainable parameters

| Component | Calculation | Parameter count |
|---|---|---:|
| Embedding table | 8 vocabulary entries × 6 values | 48 |
| Output weights | 8 scores × 6 weights | 48 |
| Output biases | One per vocabulary score | 8 |
| Total | 48 + 48 + 8 | **104** |

The script counts them with `sum(parameter.numel() for parameter in model.parameters())`.

These are adjustable numbers, not training examples. Their values are initialized when we create the model and will be updated during training.

## 8. What happens to the targets?

The target tensor y stays as integer IDs with shape (7, 4).

We do not give y to this forward method. The model produces its scores from x alone. In the training step, we will compare those scores with y to calculate an error, called a loss.

## 9. Verify with tests

In `tests/test_model.py`, we added two tests:

| Test | What it proves |
|---|---|
| Model returns scores for every token | Outputs have the expected shape, contain finite scores, stay on CPU, and yield valid predicted IDs |
| Prediction depends only on current token | An unchanged current token gets equal scores even when earlier tokens differ |

The first test uses input shape (2, 4) and output shape (2, 4, 8). The second uses input shape (2, 3) and output shape (2, 3, 8).

With all previous tests present and passing, the expected total is **13 passed**.

## 10. Installations and storage

No new installations are required. Activate the existing .venv and use PyTorch and pytest.

Model parameters and outputs live in memory. We have not saved a trained model to disk, and no training has occurred.

## What we achieved

**Input IDs → embedding lookup → linear output layer → next-token logits**

We can also select a predicted ID at each position by taking the highest score. The next training step will compare scores with correct targets and adjust the model’s parameters to reduce errors.


---

# Step 6 — Train the model

## Goal

Compare the model’s next-token scores with the correct targets, then adjust its parameters to reduce prediction errors.

For the first time in this project, the embedding table and output layer learn from our dataset. The architecture stays the same as Step 5.

## 1. What training receives

We reuse the seven input/target pairs from "hello world".

| Tensor | Shape | Meaning |
|---|---|---|
| x | (7, 4) | Input token IDs |
| y | (7, 4) | Correct next-token IDs |
| model(x) | (7, 4, 8) | Eight candidate scores at each input position |

The targets already contain the correct answers. For example, h should be followed by e. Training encourages a higher score for that correct answer relative to the alternatives.

## 2. Calculate one error number: loss

In `src/training.py`, `calculate_loss(logits, targets)` uses cross-entropy to measure how well the scores match the correct IDs.

Cross-entropy accepts raw logits. We do not apply argmax or softmax before passing them to this function.

| Expression | Purpose | Shape |
|---|---|---|
| `logits.shape[-1]` | Get vocabulary size, 8 | Python integer |
| `logits.reshape(-1, vocab_size)` | Combine examples and positions into prediction rows | (28, 8) |
| `targets.reshape(-1)` | Arrange the matching answers in the same order | (28,) |
| `F.cross_entropy(...)` | Calculate the average loss across positions | Scalar () |

The -1 asks PyTorch to infer that dimension. Seven examples times four positions gives 28 prediction rows. Reshaping reorganizes the dimensions without changing which target corresponds to each score row.

The loss is one number. Lower loss means the model assigns more probability to correct targets on average. It is not the number or percentage of wrong predictions.

## 3. Set up the optimizer

`train_model()` rejects a step count below 1, selects training mode, and creates an Adam optimizer.

| Code | Meaning |
|---|---|
| `model.train()` | Select training mode; this does not update parameters |
| `model.parameters()` | Supply the embedding and output-layer parameters |
| `torch.optim.Adam(..., lr=learning_rate)` | Create the optimizer that will adjust those parameters |

The learning rate controls the scale of adjustments. We chose 0.05 for this small demonstration; it is not a universal setting.

## 4. Measure the initial loss

Before updating anything, the function calculates `calculate_loss(model(x), y).item()` inside `torch.no_grad()`.

This records the model’s starting error. No gradient tracking is needed for this measurement.

`.item()` converts the scalar loss tensor into a Python number.

## 5. What each training iteration does

The loop repeats 200 times by default. Each iteration uses all seven examples.

| Operation | Code | What happens |
|---|---|---|
| Clear old gradients | `optimizer.zero_grad()` | Remove gradients left from the previous update |
| Forward pass | `logits = model(x)` | Run embedding lookup and the output layer |
| Measure error | `loss = calculate_loss(logits, y)` | Compare scores with correct targets |
| Backward pass | `loss.backward()` | Calculate how each parameter affects the loss |
| Update parameters | `optimizer.step()` | Use those gradients to change parameters |

PyTorch normally accumulates gradients across backward calls. Clearing them makes each update use the current iteration’s gradients.

A gradient describes how the loss changes when a parameter changes. Backpropagation calculates these gradients through the output layer and embedding lookup.

**Backward calculates gradients; the optimizer step changes parameters.** Both the embedding table and output layer can be updated. Integer input and target IDs remain unchanged.

## 6. Measure the final loss

After the final update, the function measures loss again inside `torch.no_grad()` and returns `initial_loss, final_loss`.

The expected result for this demonstration is a substantially lower final loss. Individual updates are not guaranteed to reduce loss every time.

## 7. What the runnable script does

In `scripts/step_06_training.py`, we:

1. Set a random seed for repeatable initialization in the same environment.
2. Set one CPU thread for the small workload.
3. Recreate the tokenizer, examples, tensors, and model.
4. Train for 200 updates.
5. Print the initial and final losses.
6. Use the trained model to predict a character after h.

For that final prediction:

| Tensor or operation | Shape or result |
|---|---|
| Input containing h | (1, 1) |
| Model logits | (1, 1, 8) |
| `logits[0, 0]` | (8,) |
| `argmax().item()` | One predicted token ID |
| Tokenizer decoding | Expected character: e |

`model.eval()` selects evaluation mode. `torch.no_grad()` disables gradient tracking. These are separate operations. Our current layers do not behave differently between training and evaluation modes, but using the modes prepares us for later components.

## 8. Why the loss will not reach zero

The bigram model uses only the current character. In our examples, l is followed by l, o, or d.

Every occurrence of l produces the same scores, so the model cannot confidently give a different answer for each context. It can learn a distribution of possible next characters, but it cannot use earlier characters to distinguish them.

Attention will later expand the information available for predictions.

## 9. What lower loss proves

We measure the loss on the same examples used for training. A decrease demonstrates learning those examples. It does not prove the model generalizes to unseen text.

Our seven examples are a teaching dataset, not enough data for a useful general language model.

## 10. Verify learning with tests

In `tests/test_training.py`, we check:

| Test | What it proves |
|---|---|
| Training reduces loss and updates embeddings | Final loss is below 70% of initial loss, and embedding weights changed |
| Zero training steps fail | Invalid input raises ValueError |

The first test copies the initial embedding values with `detach().clone()`. Detach removes gradient tracking from the snapshot; clone makes an independent copy so later parameter updates cannot change the saved comparison values.

Training inputs and targets have shape (7, 4), logits have shape (7, 4, 8), and embedding weights have shape (8, 6).

With all previous tests present and passing, the expected total is **15 passed**.

## 11. Installations and storage

No new installations are required. Activate the existing .venv and use PyTorch and pytest.

Training changes the model object in memory. The script does not save a checkpoint. When it exits, those learned values are not retained on disk. Running it again creates a new model and trains again.

## What we achieved

**Inputs → logits → loss → gradients → parameter updates**

Our model now learns by adjusting its embedding table and output layer. We can measure its improvement and use the trained parameters for a next-token prediction while the program runs.


---

# Step 7 — Generate text one token at a time

## Goal

Use the trained model to extend a starting text, called a **prompt**, by repeatedly predicting and appending one token.

Training changes parameters. Generation uses the learned parameters without updating them.

## 1. Prepare a trained model and a prompt

In `scripts/step_07_generation.py`, we recreate the dataset and train a fresh bigram model using Step 6’s function. We do this because we have not saved a model checkpoint yet.

We encode the prompt "h" and wrap it as one example in a tensor with shape **(1, 1)**: one example containing one token.

## 2. What the generation function does

In `src/generation.py`, `generate(model, token_ids, max_new_tokens)` validates the inputs, selects evaluation mode, and copies the prompt.

| Code | Purpose |
|---|---|
| `@torch.no_grad()` | Disable gradient tracking during generation |
| `model.eval()` | Select evaluation mode; this does not itself disable gradients |
| `generated = token_ids.clone()` | Copy the prompt so the caller’s tensor stays unchanged |
| `for _ in range(max_new_tokens)` | Run one iteration per new token |

The function rejects an empty sequence, an input without two dimensions, and a negative new-token count.

## 3. Select the last token

Each iteration uses `last_token = generated[:, -1:]`.

The colon selects all examples. The slice -1: selects the last token while retaining a sequence dimension.

For one example, last_token has shape **(1, 1)**.

Our current model is a bigram model: its predictions depend only on the current token. Passing only the last token is sufficient for this architecture. A future transformer will need preceding context too.

## 4. Calculate scores and select a token

`logits = model(last_token)` produces shape **(1, 1, 8)**: one example, one position, eight candidate next-token scores.

Then `logits[:, -1, :]` selects the final position’s scores, producing shape **(1, 8)**.

`argmax(dim=-1, keepdim=True)` selects the highest-scoring token ID for each example. Keeping the dimension gives next_token shape **(1, 1)**.

This is **greedy decoding**: always choose the highest-scoring candidate. It does not sample randomly.

## 5. Append the predicted token

`generated = torch.cat((generated, next_token), dim=1)` joins the new token onto the sequence.

| Stage | Shape |
|---|---|
| Prompt | (1, 1) |
| After one new token | (1, 2) |
| After two new tokens | (1, 3) |
| After ten new tokens | (1, 11) |

The batch dimension stays unchanged. Only sequence length increases.

For a prompt shaped **(1, 3)** and five new tokens, the result is **(1, 8)**. It is not (1, 3, 5): appending extends an existing dimension rather than creating a new one.

## 6. Repeat using the extended sequence

The appended token becomes the last token used in the following iteration. This is **autoregressive generation**: generated output becomes input for the next prediction.

After the loop, the function returns the entire sequence, including the original prompt. The script converts its IDs into a Python list and decodes them into text.

## 7. Understand your actual output

Your run produced:

| Item | Observed result |
|---|---|
| Prompt | h |
| Prompt shape | (1, 1) |
| Generated shape | (1, 11) |
| New characters | 10 |
| Prompt preserved | True |
| Parameters unchanged | True |
| Generated text | 'helo wo wo ' |

The final space is part of the generated text. The result has eleven characters: the prompt plus ten appended characters.

The model selected o after l, producing "helo" rather than "hello". It then followed the repeating transitions **o → space → w → o**.

This repetition is expected for greedy bigram generation. The same current character always produces the same scores with unchanged parameters, so greedy selection always chooses the same next token. The model cannot use a longer context to choose different continuations.

This output demonstrates learned local transitions, not understanding of words or sentences.

## 8. Verify parameters stay unchanged

Before generation, the script copies each parameter with `parameter.detach().clone()`. After generation, it compares the saved values with current parameters using `torch.equal`.

The observed result, **Parameters unchanged: True**, confirms the values did not change during generation.

The function does change the module’s mode to evaluation mode. Mode selection and parameter updates are separate concepts.

## 9. Verify generation with tests

The main test configures the output layer so token ID 2 always has the highest score. This makes expected output predictable without relying on random initialization or training.

| Test | What it verifies |
|---|---|
| Appending selected tokens | Prompt [3, 4] plus three predictions becomes [3, 4, 2, 2, 2]; prompt and parameters are preserved |
| Zero new tokens | Return the prompt values without extending them |
| Empty prompt | Raise ValueError |
| Negative token count | Raise ValueError |

You reported **19 passed in 3.20 seconds**, including all earlier tests.

## 10. Installations and storage

No new installations are required. Activate the existing .venv and use PyTorch and pytest.

The script trains and generates within one process. Model parameters and generated tensors remain in memory during execution. No checkpoint or generated-text file is saved by this code.

## What we achieved

**Prompt → last-token scores → highest-scoring ID → append → repeat → decode**

Our trained model can now extend text one token at a time. Generation preserves the prompt and uses learned parameters without updating them. Its limited context explains why the output can repeat.


---

# Step 8 — Causal self-attention

## Goal

Let each token combine information from itself and preceding token positions. Our bigram model processes each token independently; attention provides a way to build representations using context.

This step builds and tests a standalone attention component. It does not change the existing bigram model or train a transformer.

## 1. Which positions can contribute?

For "hell", causal attention permits:

| Receiving position | Character | Allowed contributing positions |
|---|---|---|
| 0 | h | 0 |
| 1 | e | 0, 1 |
| 2 | l | 0, 1, 2 |
| 3 | l | 0, 1, 2, 3 |

Future positions are forbidden. During next-token training, the next character is already present at a later input position, so unrestricted attention could reveal the correct answer. Causal masking prevents this information leak.

## 2. Create queries, keys, and values

In `src/attention.py`, `CausalSelfAttention` contains three trainable linear layers, each mapping six input features to six output features without bias.

| Vector | Role |
|---|---|
| Query Q | Compared with keys to score contributing positions |
| Key K | Provides the representation compared with each query |
| Value V | Supplies the information mixed into the output |

In forward(), `q = self.query(x)`, `k = self.key(x)`, and `v = self.value(x)` apply those layers to the same input embeddings.

This is self-attention because Q, K, and V all come from the same sequence.

Each layer has 6 × 6 weights, so this standalone attention component has 108 trainable parameters.

## 3. Calculate pairwise scores

`scores = q @ k.transpose(-2, -1)` computes a dot product between every query and every key within each example.

For input shape (1, 4, 6):

| Tensor | Shape |
|---|---|
| Q, K, V | (1, 4, 6) |
| Transposed K | (1, 6, 4) |
| Scores | (1, 4, 4) |

Rows identify receiving query positions; columns identify contributing key positions. A score compares one receiving position with one potential source position.

The code then divides scores by the square root of the query dimension, here sqrt(6). This scaling controls score magnitude as the vector dimension changes.

## 4. Build and apply the causal mask

`torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device)` creates a square Boolean matrix.

`torch.tril(...)` keeps the lower triangle, including the diagonal:

| Query position | Key 0 | Key 1 | Key 2 | Key 3 |
|---|---|---|---|---|
| 0 | Allowed | Blocked | Blocked | Blocked |
| 1 | Allowed | Allowed | Blocked | Blocked |
| 2 | Allowed | Allowed | Allowed | Blocked |
| 3 | Allowed | Allowed | Allowed | Allowed |

The mask has shape (4, 4) and is applied across examples.

`scores.masked_fill(~allowed, float("-inf"))` replaces forbidden scores with negative infinity. The tilde inverts the Boolean mask, selecting blocked positions.

## 5. Convert scores into attention weights

`weights = torch.softmax(scores, dim=-1)` operates across the contributing positions in each row.

The resulting weights are nonnegative, sum to approximately one per row, and are zero for masked future positions.

These are weights over **sequence positions**, not probabilities over vocabulary tokens.

## 6. Mix the value vectors

`output = weights @ v` computes a weighted sum of value vectors for each receiving position.

| Tensor | Shape |
|---|---|
| Attention weights | (1, 4, 4) |
| Value vectors | (1, 4, 6) |
| Output vectors | (1, 4, 6) |

The output contains a six-value vector per position, combining information from allowed positions. It does not contain next-token vocabulary scores.

The component returns both output and weights, allowing us to inspect how the combination was formed.

## 7. Why the demonstration uses uniform weights

In `scripts/step_08_attention.py`, we embed "hell" and temporarily set the query and key weight matrices to zero.

All unmasked scores become zero, so softmax assigns equal weights to the allowed positions:

| Receiving position | Weight for position 0 | Position 1 | Position 2 | Position 3 |
|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 |
| 1 | 1/2 | 1/2 | 0 | 0 |
| 2 | 1/3 | 1/3 | 1/3 | 0 |
| 3 | 1/4 | 1/4 | 1/4 | 1/4 |

The value layer remains randomly initialized. In this controlled setup, each output is simply the average of its allowed value vectors.

This manual zeroing is for inspection, not a recommended training initialization. Normally, the trainable Q and K layers can produce different scores for different query/key pairs.

## 8. Read the script’s expected shapes

| Item | Shape |
|---|---|
| Token IDs for "hell" | (1, 4) |
| Embeddings | (1, 4, 6) |
| Attention output | (1, 4, 6) |
| Attention weights | (1, 4, 4) |
| Displayed weights[0] | (4, 4) |

The script expects both checks to be True: rows sum to one, and future positions have zero weight.

## 9. Verify the behavior with tests

In `tests/test_attention.py`, we added two tests:

| Test | What it verifies |
|---|---|
| Future information is blocked | Changing the last input vector does not change earlier output vectors; future weights are zero and rows sum to one |
| Uniform attention averages allowed values | With zero Q and K weights, each output equals the mean of its allowed value vectors |

The tests use `torch.allclose` for floating-point comparisons rather than requiring exact equality for calculated averages.

With all earlier tests present and passing, the expected total is **21 passed**. This is an expected result, not a reported test run.

## 10. What attention does and does not provide yet

The two l positions can now receive different context-dependent outputs because their allowed contributing positions differ.

We have built one attention head. We have not yet added explicit position representations, a feed-forward network, residual connections, normalization, or integration with the language-model output layer.

The causal mask supplies a direction restriction; it is not a replacement for position representations.

## 11. Installations and storage

No new installations are required. Activate the existing .venv and use PyTorch and pytest.

The demonstration runs without gradient tracking and performs no training updates. Parameters and output tensors exist in memory during execution; the script saves no checkpoint.

## What we achieved

**Embeddings → Q/K/V → scaled scores → causal mask → softmax weights → weighted value vectors**

We now have a tested component that lets each position gather information from itself and earlier positions while blocking access to future input.


---

# Step 9 — Build a transformer block

## Goal

Combine Step 8’s causal attention with a feed-forward network, layer normalization, and residual connections.

The block receives vectors, processes context and features, and returns vectors of the same shape. It is still a component, not a complete language model.

## 1. Why attention alone is not the whole block

Attention mixes information between allowed token positions. A transformer block also transforms the features within each position and provides direct paths for information and gradients.

| Component | Purpose |
|---|---|
| Causal attention | Gather information from the current and preceding positions |
| Feed-forward network | Transform each position’s features independently |
| Layer normalization | Normalize features within each token vector |
| Residual addition | Add a processing result back to its input |

We retain our single attention head as a simplified implementation.

## 2. What the class creates

In `src/transformer.py`, `TransformerBlock` inherits from `nn.Module` and creates:

- `norm1`: layer normalization before attention.
- `attention`: the causal component from Step 8.
- `norm2`: layer normalization before the feed-forward network.
- `feed_forward`: two linear layers with GELU between them.

Because normalization comes before each processing branch, this is a **pre-normalization block**.

## 3. Normalize each token’s features

`nn.LayerNorm(embedding_dim)` operates across the last dimension. With input shape (1, 4, 6), it normalizes each six-feature vector independently.

It does not normalize across sequence positions or mix examples. It also has trainable scale and bias values for the features.

Normalization helps control the numerical scale of representations. Its output retains shape (1, 4, 6).

## 4. Apply attention and the first residual connection

The first forward line is `attention_output, _ = self.attention(self.norm1(x))`.

The steps are:

1. Normalize x.
2. Pass normalized vectors into causal attention.
3. Receive context-dependent vectors and attention weights.
4. Keep the vectors; the underscore means we do not use the weights here.

Then `x = x + attention_output` adds the attention result to the original, unnormalized x.

| Tensor | Shape |
|---|---|
| Original x | (1, 4, 6) |
| Normalized attention input | (1, 4, 6) |
| Attention output | (1, 4, 6) |
| First residual result | (1, 4, 6) |

The addition is element by element. It preserves a direct input path while adding contextual information.

## 5. Transform features using the feed-forward network

The network uses the same weights at every token position. It does not itself exchange information between positions.

| Operation | Feature change | Tensor shape |
|---|---|---|
| First linear layer | 6 → 24 | (1, 4, 24) |
| GELU | 24 → 24 | (1, 4, 24) |
| Second linear layer | 24 → 6 | (1, 4, 6) |

GELU adds nonlinearity. Without a nonlinear operation, the two linear layers could be represented as a single affine transformation.

The expansion factor of four is our architecture choice, not a requirement.

## 6. Apply the second residual connection

`feed_forward_output = self.feed_forward(self.norm2(x))` normalizes the updated x and transforms its features.

Then `x = x + feed_forward_output` adds the result to that updated representation.

Returning to six features makes the branch output match x for residual addition. It does not convert vectors into token IDs or set the vocabulary size.

The block finally returns x.

## 7. Follow the complete computation

| Stage | Shape |
|---|---|
| Input vectors | (1, 4, 6) |
| Normalize and attend | (1, 4, 6) |
| Add original input | (1, 4, 6) |
| Normalize and expand | (1, 4, 24) |
| GELU and reduce | (1, 4, 6) |
| Add second residual input | (1, 4, 6) |

Preserving shape lets us pass the output into another compatible block later.

## 8. What the demonstration checks

In `scripts/step_09_transformer.py`, we embed "hell", run the block, and check that its output is finite.

Then we copy the input vectors and change one feature of the final position. We run the block again and compare earlier positions.

Expected results:

| Printed item | Expected value |
|---|---|
| Token IDs shape | (1, 4) |
| Block input shape | (1, 4, 6) |
| Block output shape | (1, 4, 6) |
| Output is finite | True |
| Earlier positions unchanged | True |

Earlier positions remain unchanged because attention is causal, and normalization and feed-forward processing operate within each position.

The script runs without gradient tracking and performs no training updates.

## 9. Verify the block with tests

In `tests/test_transformer.py`, we added three tests:

| Test | What it proves |
|---|---|
| Shape and causality | Output shape is preserved, values are finite, and changing the future does not affect earlier outputs |
| Zero processing branches | When both attention and feed-forward branches output zero, residual connections return the input unchanged |
| Gradient flow | Backpropagation produces finite gradients for input vectors and all block parameters |

The gradient test creates input with `requires_grad=True`, calculates a scalar mean-squared-output loss, and calls backward(). This tests differentiability; it does not run an optimizer or train a language model.

With all earlier tests present and passing, the expected total is **24 passed**. This is the expected outcome, not a reported test run.

## 10. Installations and storage

No new installations are required. Activate the existing project .venv and use PyTorch and pytest.

The block and tensors remain in memory during execution. We have not saved a checkpoint.

## What we achieved

A working transformer block combines **causal attention, feature transformation, normalization, and residual paths** while preserving shape.

We have not yet connected this block to token and position embeddings and a vocabulary output layer. That assembly belongs to Step 10.


---

# Step 10 — Assemble a tiny GPT-style model

## Goal

Combine our components into a complete, simplified transformer language model, then train it and generate text.

The model has one causal attention head and one transformer block. It demonstrates GPT-style next-token prediction on a tiny character dataset; it is not a production LLM.

## 1. What the model contains

In `src/gpt.py`, `TinyGPT` inherits from nn.Module and assembles:

| Component | Purpose |
|---|---|
| TokenEmbedding | Look up the vector for each character ID |
| Position embedding | Look up a vector for each position in the context window |
| TransformerBlock | Combine allowed context and transform features |
| Final LayerNorm | Normalize each final token representation |
| Linear output layer | Convert representations into vocabulary logits |

The old bigram model remains available for comparison.

## 2. Why add position embeddings?

Token embeddings identify which character is present. Position embeddings give an explicit trainable representation of where it occurs within the input window.

For block_size 4 and embedding_dim 6, the position table has shape (4, 6): one six-value vector for each position 0 through 3.

The causal mask controls which positions may contribute. Position embeddings supply explicit position information; these are different roles.

## 3. Follow the forward method

The method validates that token_ids has two dimensions and that sequence length is between 1 and block_size.

| Code | Purpose |
|---|---|
| `torch.arange(sequence_length, device=token_ids.device)` | Create position IDs from zero to sequence length minus one |
| `self.token_embedding(token_ids)` | Look up character vectors |
| `self.position_embedding(positions)` | Look up position vectors |
| `token_vectors + position_vectors` | Combine identity and position by elementwise addition |
| `self.block(x)` | Apply causal attention and feed-forward processing with residual paths |
| `self.final_norm(x)` | Normalize the final features |
| `self.output(x)` | Produce scores for every vocabulary token |

Position vectors are broadcast across examples. Every example uses the same position table, while its character IDs may differ.

## 4. Understand the shapes

For seven examples of four tokens:

| Stage | Shape |
|---|---|
| Token IDs | (7, 4) |
| Position IDs | (4,) |
| Token vectors | (7, 4, 6) |
| Position vectors | (4, 6) |
| Combined representations | (7, 4, 6) |
| Transformer and final normalization output | (7, 4, 6) |
| Vocabulary logits | (7, 4, 8) |

The output has one score per vocabulary entry at every input position. Position addition does not concatenate vectors or increase their length.

## 5. How training reuses earlier work

In `scripts/step_10_gpt.py`, we recreate the tokenizer and seven examples, then create a fresh TinyGPT model.

We reuse calculate_loss() from Step 6 because the new model returns the same logits format. The optimizer now receives the parameters of the transformer model, including the position embeddings.

Each of 400 updates performs:

1. Clear previous gradients.
2. Run the model on x.
3. Compare its logits with y using cross-entropy.
4. Calculate gradients with backward().
5. Update parameters with optimizer.step().

We choose learning rate 0.01 for this demonstration and use one CPU thread.

Initial and final loss measurements run without gradient tracking. The expected final training loss is substantially lower; exact numbers depend on the environment.

## 6. Count the parameters

For vocabulary size 8, embedding dimension 6, and block size 4:

| Component | Parameters |
|---|---:|
| Token embeddings: 8 × 6 | 48 |
| Position embeddings: 4 × 6 | 24 |
| Query, key, value weights: 3 × 6 × 6 | 108 |
| Two block LayerNorms: 2 × (6 scales + 6 biases) | 24 |
| Feed-forward expansion: 24 × 6 weights + 24 biases | 168 |
| Feed-forward reduction: 6 × 24 weights + 6 biases | 150 |
| Final LayerNorm: 6 scales + 6 biases | 12 |
| Vocabulary output: 8 × 6 weights + 8 biases | 56 |
| **Total** | **590** |

These parameters are adjustable values. Their count is independent of the number of training examples.

## 7. Why generation must use context now

Step 7’s generation function passed only the last token because the bigram model used only that token.

TinyGPT’s generate() method instead uses `context = generated[:, -self.block_size:]`. It passes the most recent available tokens, up to four, through the transformer.

| Stored sequence length | Context length passed to the model |
|---|---|
| 1 | 1 |
| 3 | 3 |
| 4 | 4 |
| 9 | 4: the last four tokens |

Passing only the last token would discard the preceding context that attention can use.

The stored sequence keeps growing. Cropping limits what the model reads; it does not delete the earlier tokens from the returned sequence.

## 8. Follow one generation iteration

1. Crop the most recent context.
2. Run the model on that context.
3. Select logits for the final context position.
4. Choose the highest-scoring ID using argmax.
5. Append that ID to generated.

For one example and context length T:

| Tensor | Shape |
|---|---|
| Context | (1, T), where 1 ≤ T ≤ 4 |
| Logits | (1, T, 8) |
| Final-position logits | (1, 8) |
| Selected next token | (1, 1) |

The position IDs restart at zero for each cropped window. Generation uses evaluation mode and disables gradient tracking. No optimizer updates occur.

A one-token prompt plus ten new tokens returns shape (1, 11).

## 9. Understand the expected output and limitations

The script should report input shape (7, 4), logits shape (7, 4, 8), position table shape (4, 6), parameter count 590, decreased training loss, and generated shape (1, 11).

The generated string should contain eleven characters beginning with h. Exact text can vary.

Even with a transformer, this dataset is extremely small and contains overlapping windows and ambiguous short contexts. Lower training loss is not evidence of general language understanding or reliable performance on unseen text.

## 10. Verify the assembled architecture

In `tests/test_gpt.py`, we added three tests:

| Test | What it proves |
|---|---|
| Causality | Changing the final input ID cannot change earlier logits |
| Position embeddings affect output | Changing one position vector affects that position’s output while an earlier position stays unchanged |
| Generation crops context | A prompt longer than the window can be extended without passing an oversized input into forward(); the original prompt is preserved |

The generation test fixes the output layer to always choose token ID 2, making the expected appended sequence deterministic.

With all earlier tests present and passing, the expected total is **27 passed**. This is an expected result, not a reported test run.

## 11. Installations and storage

No new installations are required. Activate the existing project .venv.

The script creates, trains, and uses the model in one process. Parameters and outputs live in memory. No checkpoint is saved, so running the script again creates a fresh model and trains it again.

## What we achieved

**Token IDs + positions → embeddings → transformer block → final normalization → vocabulary logits → next-token selection**

Our project now has a complete tiny GPT-style model with training and context-aware generation. Step 11 will trace how these components execute during generation.

