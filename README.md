# mini-GPT

Lightweight implementation of a GPT model for character-level text generation.

This project implements a compact and efficient **transformer-based language model** for character-level generation. The core logic is defined in **`train.py`**, featuring **multi-head self-attention** and **Rotary Position Embeddings (RoPE)** for improved positional encoding.

Model experimentation and architectural modifications are developed in **`gpt-dev.ipynb`**, which serves as the foundation for the finalized training script. The **`data/`** folder includes an **`input.txt`** file for localized testing, with optional support for a subset of **Hugging Face’s American Stories** dataset (currently excluded from version control).

Performance and speed comparisons between model variants are documented in **`benchmarking.py`**, and also visualized within the notebook. This project supports both **GPU and CPU** execution environments. 

This project remins under **active development**.


## Features

- Transformer-based language model with multi-head attention
- Rotary Position Embeddings (RoPE) for better position encoding
- Feed-forward networks with GELU activation
- Layer Normalization
- Residual Connections
- Dropout for regularization
- Configurable model architecture (embedding dimension, number of heads, layers)
- Training with early stopping and validation monitoring
- Character-level text generation with temperature sampling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shash23/mini-gpt.git
cd mini-gpt
```

2. Install the required dependencies:
```bash
pip install torch numpy matplotlib
```

## Usage

### Training the Model

1. Prepare your training data:
   - Place your text data in `input.txt` in the project root directory
   - Or use the default sample text that will be created if no input file is found

2. Run the training script:
```bash
python train.py
```

The model will train with the following default hyperparameters:
- Batch size: 16
- Block size: 32
- Embedding dimension: 128
- Number of attention heads: 4
- Number of transformer layers: 4
- Dropout: 0.2
- Learning rate: 1e-3

### Generating Text

After training, the model will automatically generate sample text. It will also display loss. You can also generate text programmatically:

```python
import torch
from train import TransformerLanguageModel, decode

# Load your trained model
model = TransformerLanguageModel()
model.load_state_dict(torch.load('best_model.pth'))

# Generate text
context = torch.zeros((1, 1), dtype=torch.long)
generated_text = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
print(generated_text)
```

## Citations

This project builds on foundational work in transformers and rotary position embeddings. Key references:

- **[Andrej Karpathy - Let's Build GPT from Scratch (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)**  
  A hands-on walkthrough explaining transformer mechanics and GPT architecture by Andrej Karpathy.  
  *Let's build GPT: from scratch, in code, spelled out.* (2023)

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**  
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017).  
  Introduced the Transformer architecture, replacing recurrence with multi-head self-attention.

- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)**  
  Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu (2021).  
  Proposed Rotary Position Embeddings (RoPE), which use rotation matrices to encode position and enable relative positioning within self-attention mechanisms.
