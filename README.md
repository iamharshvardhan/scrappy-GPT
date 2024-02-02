# GPT-from-scratch | Bigram Model for Character-Level Text Generation

This project implements a bigram model for character-level text generation using PyTorch. The bigram model predicts the next character in a sequence based on the current character.

The [data](./data/input.txt) used here to train the neural net is a Shakesperean play "Coriolanus" from Act 1, Scene 1 to Act 2, Scene 2.

## Usage

To use this script:

1. Ensure you have PyTorch installed. If not, you can install it via pip: `pip install torch`.

2. Update the hyperparameters according to your requirements. The hyperparameters include:
   - `batch_size`: Batch size for training.
   - `block_size`: Size of the sequence block.
   - `max_iters`: Maximum number of iterations for training.
   - `eval_internals`: Interval for evaluating the loss during training.
   - `learning_rate`: Learning rate for the optimizer.
   - `n_embd`: Dimensionality of the embedding.
   - `n_head`: Number of attention heads.
   - `n_layer`: Number of transformer layers.
   - `dropout`: Dropout probability.

3. Run the script. It will train the bigram model on the input text data and generate text based on the trained model.

## Description

The script consists of the following components:

- Data Loading: Reads the input text file, encodes characters, and prepares the dataset.
- Model Components:
  - `Head`: Represents one head of self-attention.
  - `MultiHeadAttention`: Consists of multiple heads of self-attention in parallel.
  - `FeedForward`: Implements the feedforward network.
  - `Block`: Represents a transformer block consisting of communication followed by computation.
  - `BigramLanguageModel`: Implements the bigram language model.
- Training Loop: Trains the bigram model using stochastic gradient descent.
- Text Generation: Generates text based on the trained bigram model.

## Dependencies

- Python 3.x
- PyTorch

## Example

```python
# Run the script
python bigram_model.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---
Feel free to customize and extend the script according to your specific needs! If you have any questions or encounter any issues, please don't hesitate to contact the author.
