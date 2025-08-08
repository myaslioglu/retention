# Transformer Encoder Implementation

A clean, educational implementation of the Transformer Encoder architecture from the "Attention Is All You Need" paper, built with PyTorch.

## 🚀 Features

- **Complete Transformer Encoder**: 6-layer encoder with multi-head self-attention
- **Modular Design**: Clean separation of components for easy understanding and modification
- **Configurable Architecture**: TOML-based configuration for easy experimentation
- **Educational Focus**: Well-documented code with clear variable names and comprehensive docstrings
- **Production Ready**: Proper error handling, logging, and CLI interface
- **Flexible Tokenization**: Support for both tiktoken (GPT-2 style) and custom tokenizers

## 📋 Architecture Overview

### Key Components

- **InputEmbeddings**: Token embedding layer with learnable parameters
- **PositionalEncoding**: Sinusoidal positional encodings with dropout
- **SelfAttention**: Single attention head with optional causal masking
- **MultiHeadAttention**: Parallel attention heads with output projection
- **ResidualAddNorm**: Residual connections followed by layer normalization
- **FeedForward**: Two-layer MLP with ReLU activation and dropout
- **TransformerEncoderLayer**: Complete encoder layer combining all components
- **TransformerEncoder**: Full encoder stack with embeddings and N layers

### Default Configuration

```toml
# Model architecture
vocab_size = 50257        # GPT-2 style vocabulary
hidden_size = 512         # Model dimension
seq_len = 1024           # Maximum sequence length
n_heads = 8              # Number of attention heads
ff_hidden_size = 2048    # Feed-forward hidden dimension
n_layers = 6             # Number of encoder layers
dropout_pe = 0.1         # Dropout probability

# Training
batch_size = 32
learning_rate = 1e-4
max_epochs = 100
```

**Parameters**: ~85M (similar to GPT-2 Small)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/MayukhSobo/Transformer.git
cd Transformer

# Install dependencies
pip install torch tiktoken toml

# Or using requirements.txt
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage

```python
from transformer.encoder import TransformerEncoder
import torch

# Create model
model = TransformerEncoder(
    vocab_size=50257,
    hidden_size=512,
    seq_len=1024,
    dropout_pe=0.1,
    n_heads=8,
    ff_hidden_size=2048,
    n_layers=6
)

# Forward pass
token_ids = torch.randint(0, 50257, (32, 128))  # [batch, seq_len]
output = model(token_ids)  # [batch, seq_len, hidden_size]
```

### Configuration-Based Usage

```python
import toml
from transformer.encoder import TransformerEncoder

# Load configuration
config = toml.load('config/model_config.toml')

# Create model from config
model = TransformerEncoder(**config['model'])
```

### Command Line Interface

```bash
# Train model with default config
python main.py

# Train with custom config
python main.py --config path/to/config.toml

# Training with custom parameters
python main.py --batch_size 64 --learning_rate 5e-4
```

## 📁 Project Structure

```
transformer-encoder/
├── transformer/
│   ├── __init__.py
│   ├── attention.py          # Self-attention and multi-head attention
│   ├── embedding.py          # Token embeddings and positional encoding
│   ├── encoder.py            # Encoder layers and full encoder
│   ├── feed_forward.py       # Feed-forward network
│   └── normalization.py      # Residual connections and layer norm
├── data/
│   ├── __init__.py
│   └── dataset.py           # TinyStory dataset implementation
├── config/
│   └── model_config.toml    # Model and training configuration
├── main.py                  # Training script with CLI
├── requirements.txt
└── README.md
```

## 🏗️ Architecture Details

### Multi-Head Self-Attention

- **8 attention heads** with 64 dimensions each (512 total)
- **Scaled dot-product attention** with optional causal masking
- **Separate Q, K, V projections** for each head
- **Output projection** to combine all heads

### Feed-Forward Network

- **Two linear layers**: 512 → 2048 → 512
- **ReLU activation** between layers
- **Dropout regularization** after activation

### Residual Connections & Normalization

- **Pre-norm architecture**: LayerNorm applied before sub-layers
- **Residual connections** around attention and feed-forward blocks
- **Separate normalization** for attention and FFN paths

### Positional Encoding

- **Sinusoidal encoding** following the original paper
- **Learnable parameters**: Optional elementwise affine transformation
- **Dropout regularization** applied after position embedding addition

## 🎯 Educational Focus

This implementation prioritizes:

- **Clarity over performance**: Code is written to be easily understood
- **Comprehensive documentation**: Every module and function is well-documented
- **Modular design**: Each component can be studied and modified independently
- **Type hints**: Full type annotations for better code understanding
- **Educational comments**: Inline explanations of key concepts

## 🔧 Customization

### Creating Custom Attention Patterns

```python
# Encoder attention (bidirectional)
encoder_attention = SelfAttention(
    hidden_size=512,
    max_seq_len=1024,
    d_k=64,
    dropout_pe=0.1,
    masking=False  # No causal masking
)

# Decoder attention (causal)
decoder_attention = SelfAttention(
    hidden_size=512,
    max_seq_len=1024,
    d_k=64,
    dropout_pe=0.1,
    masking=True  # Causal masking enabled
)
```

### Adjusting Model Size

```python
# Smaller model (similar to GPT-2 micro)
small_model = TransformerEncoder(
    hidden_size=256,
    n_heads=4,
    ff_hidden_size=1024,
    n_layers=4
)

# Larger model (similar to GPT-2 medium)
large_model = TransformerEncoder(
    hidden_size=1024,
    n_heads=16,
    ff_hidden_size=4096,
    n_layers=12
)
```

## 📊 Training

The implementation includes:

- **TinyStory dataset** support for quick experimentation
- **Configurable training loop** with proper loss computation
- **Gradient clipping** and learning rate scheduling
- **Model checkpointing** and logging

## 🤝 Contributing

This is an educational implementation. Contributions that improve clarity, add documentation, or fix bugs are welcome!

## 📚 References

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation guide

## 📄 License

MIT License - Feel free to use this code for educational purposes.

---

**Note**: This implementation focuses on the encoder portion of the Transformer. For sequence-to-sequence tasks, you would need to implement the decoder as well.
