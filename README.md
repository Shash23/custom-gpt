# mini-GPT

Lightweight implementation of a GPT model for character-level text generation.

This project implements a compact and efficient **transformer-based language model** for character-level generation. The core logic is defined in **`train.py`**, featuring **multi-head self-attention** and **Rotary Position Embeddings (RoPE)** for improved positional encoding.

All model development occurs in the **`development/`** directory. Model experimentation and architectural modifications are developed in **`gpt-dev.ipynb`**, which serves as the foundation for the finalized training script. Performance and speed comparisons between model variants are documented in **`benchmarking.py`**, and also visualized within the notebook.

The **`data/`** directory includes an **`input.txt`** file for localized testing, with optional support for a subset of **Hugging Face's American Stories** dataset (currently excluded from version control).

The **`notes/`** directory includes notes I took during learning and development of the project.

This project supports both **GPU and CPU** execution environments. The model will automatically use CUDA if available, falling back to CPU if not. Training times will vary significantly between GPU and CPU execution.

This project remains under **active development**.

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

## Project Structure

```
mini-gpt/
├── data/                  # Data directory
│   ├── input.txt         # Training text data
│   └── american_stories/ # Optional dataset (excluded from git)
├── development/          # Core development files
│   ├── train.py         # Main training script
│   ├── gpt-dev.ipynb    # Development notebook
│   └── benchmarking.py  # Performance comparisons
├── notes/               # Development notes
└── README.md           # This file
```

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
   - Place your text data in `data/input.txt`
   - Or use the default sample text that will be created if no input file is found

2. Run the training script:
```bash
cd development
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

The trained model will be saved as `development/best_model.pth`. This checkpoint contains the model's state dictionary and can be used for inference or continued training.

### Development Environment

For development and experimentation:
1. The `gpt-dev.ipynb` notebook contains interactive development and visualization
2. Use `benchmarking.py` to compare performance between different model variants
3. All development files are in the `development/` directory for easy access

### Generating Text

After training, the model will automatically generate sample text. It will also display loss. You can also generate text programmatically:

```python
import torch
import sys
import os

# Add the development directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'development'))
from train import TransformerLanguageModel, decode

# Load your trained model
model = TransformerLanguageModel()
model.load_state_dict(torch.load('development/best_model.pth'))

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
