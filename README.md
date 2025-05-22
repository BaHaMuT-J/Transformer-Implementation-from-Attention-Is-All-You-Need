# Transformer Implementation from _Attention Is All You Need_

## Overview

This repository is a personal project aimed at deepening my understanding of the Transformer architecture as introduced in the paper _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_. It features a complete implementation of the original Encoder-Decoder model architecture, which has since become foundational in modern NLP applications.

Inspired by Andrej Karpathy's `nanogpt` and `minbpe`, I extended his Decoder-Only implementation to build a full Transformer with both Encoder and Decoder components. I trained and evaluated the models on different datasets, including the CNN/DailyMail summarization dataset and Shakespeare text.

## Transformer Architecture

The full implementation is available in [Transformer.py](./Transformer.py). The following core components are implemented:

- **`Head`**: Implements a single self-attention head.
- **`MultiHeadAttention`**: Combines multiple attention heads into multi-head self-attention.
- **`FeedForward`**: Implements a position-wise feedforward neural network.
- **`SubLayer`**: Applies residual connection followed by layer normalization (used for attention and feedforward layers).
- **`Encoder`**: Composed of one `MultiHeadAttention` and one `FeedForward` layer, following the architecture from the original paper.
- **`Decoder`**: Contains two `MultiHeadAttention` layers and one `FeedForward` layer, as described in the original Transformer paper.
- **`PositionalEncoding`**: Computes fixed positional encodings to inject order information into input tokens.
- **`Transformer_Decoder`**: A Decoder-Only Transformer model, based on Karpathy's `nanogpt` but with modifications.
- **`Transformer_Encoder_Decoder`**: A full Encoder-Decoder Transformer model implementation.

## Training & Results

### 1. [CNN/DailyMail Dataset](./Transformer_CNN_Training.ipynb)

| **Hyperparameter**        | **Value** |
| ------------------------- | --------- |
| Vocabulary Size           | 1024      |
| Max Input Length          | 512       |
| Max Target Length         | 128       |
| Number of Layers          | 6         |
| Number of Attention Heads | 8         |
| Embedding Dimension       | 512       |
| Dropout Rate              | 0.1       |
| Learning Rate             | 0.001     |
| Epochs                    | 10        |
| Batch Size                | 32        |

**Results**:

- Train Loss: 5.9204
- Validation Loss: 6.6835
- Test Loss: 6.6941

> Due to hardware limitations, training was conducted on a subset of the dataset: the first 1,000 samples for tokenizer training, and 10,000/2,500/2,500 samples for train/validation/test respectively. The model struggled with this task and generated largely unreadable summaries.

---

### 2. [Shakespeare Text (BPE Tokenizer)](./Transformer_Shakespeare_Training.ipynb)

| **Hyperparameter**        | **Value** |
| ------------------------- | --------- |
| Vocabulary Size           | 1024      |
| Max Input Length          | 256       |
| Number of Layers          | 6         |
| Number of Attention Heads | 6         |
| Embedding Dimension       | 384       |
| Dropout Rate              | 0.2       |
| Learning Rate             | 0.0001    |
| Epochs                    | 1000      |
| Batch Size                | 64        |

**Results**:

- Train Loss: 4.0634
- Validation Loss: 4.4151
- Test Loss: 4.6359

> Performance improved compared to the CNN dataset. However, results were still not on par with Karpathy's `nanogpt` example.

---

### 3. [Shakespeare Text (Character Tokenizer)](./Transformer_Shakespeare_Training_Char_Tokenizer.ipynb)

| **Hyperparameter**        | **Value** |
| ------------------------- | --------- |
| Vocabulary Size           | 1024      |
| Max Input Length          | 256       |
| Number of Layers          | 6         |
| Number of Attention Heads | 6         |
| Embedding Dimension       | 384       |
| Dropout Rate              | 0.2       |
| Learning Rate             | 0.0003    |
| Epochs                    | 1000      |
| Batch Size                | 64        |

**Results**:

- Train Loss: 1.7108
- Validation Loss: 1.7829

> By switching to a character-level tokenizer (similar to the one used by Karpathy), the model's performance improved significantly and came closer to expected benchmarks.

## License

This project is based on Andrej Karpathy's [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture/tree/master) and [minbpe](https://github.com/karpathy/minbpe/tree/master), licensed under the MIT License.

The original license of minbpe is included [here](./LICENSE).

## Citation

This project builds on the ideas introduced in the following paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). _Attention Is All You Need_. arXiv:1706.03762. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
