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

