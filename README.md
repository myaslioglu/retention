# [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

A complete, educational implementation of the Transformer architecture from the "Attention Is All You Need" paper, built with PyTorch. This implementation includes both encoder and decoder components with a modular, well-documented design.

## 🚀 Features

- **Complete Transformer Architecture**: Full encoder-decoder implementation
- **Modular Attention System**: Separate self-attention, multi-head attention, and cross-attention modules
- **Configurable Architecture**: TOML-based configuration for easy experimentation
- **Educational Focus**: Well-documented code with comprehensive docstrings
- **Production Ready**: Proper error handling, logging, and CLI interface
- **Flexible Tokenization**: Support for both tiktoken (GPT-2 style) and custom tokenizers
- **TinyStories Dataset**: Integrated dataset loading and preprocessing

## 📋 Architecture Overview

### Core Components

#### Attention Mechanisms
- **SelfAttention**: Single attention head with optional causal masking
- **MultiHeadAttention**: Parallel attention heads with output projection
- **CrossAttention**: Cross-attention for encoder-decoder interaction (fully implemented)

#### Encoder Stack
- **EncoderLayer**: Self-attention + feed-forward with residual connections
- **Encoder**: Complete encoder with embeddings, positional encoding, and N layers

#### Decoder Stack
- **DecoderLayer**: Masked self-attention + cross-attention + feed-forward
- **Decoder**: Complete decoder with embeddings, positional encoding, and N layers

#### Supporting Modules
- **Embeddings**: Token embedding layer with learnable parameters
- **PositionalEncoding**: Sinusoidal positional encodings with dropout
- **ResidualAddNorm**: Residual connections followed by layer normalization
- **FeedForward**: Two-layer MLP with ReLU activation and dropout

### Default Configuration

```toml
[model]
hidden_size = 512         # Model dimension
seq_len = 1024           # Maximum sequence length
vocab_size = 30000       # Vocabulary size
dropout_pe = 0.1         # Dropout probability
n_heads = 8              # Number of attention heads
ff_hidden_size = 2048    # Feed-forward hidden dimension
n_layers = 6             # Number of encoder/decoder layers

[tokenizer]
kind = 'tiktoken'        # tiktoken | custom
model = 'gpt-2'          # only applicable for tiktoken tokenizer

[training]
batch_size = 32
epochs = 10
learning_rate = 0.0005

[dataset]
path = "./data/TinyStories.txt"
```

**Parameters**: ~85M (similar to GPT-2 Small)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/MayukhSobo/Transformer.git
cd Transformer

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install torch tiktoken python-box click datasets codetiming tqdm
```

## 📖 Usage

### Basic Usage

```python
from transformer.encoder.model import get_encoder
from transformer.decoder.model import get_decoder
from config import Config
import torch

# Load configuration
from pathlib import Path
config = Config(Path("config.toml"))

# Create models
encoder = get_encoder(config)
decoder = get_decoder(config)

# Forward pass
token_ids = torch.randint(0, 50257, (32, 128))  # [batch, seq_len]
encoder_output = encoder(token_ids)             # [batch, seq_len, hidden_size]
decoder_output = decoder(token_ids, encoder_output)  # [batch, seq_len, hidden_size]
```

### Training Pipeline

```python
from model import create_model

# Train model with configuration
from pathlib import Path
create_model(Path("config.toml"))
```

### Command Line Interface

```bash
# Train model with default config
python main.py

# Train with custom config
python main.py --config path/to/config.toml
```

## 📁 Project Structure

```
Transformer/
├── transformer/                 # Core transformer modules
│   ├── __init__.py
│   ├── attentions/             # Attention mechanisms
│   │   ├── __init__.py
│   │   ├── self.py             # Self-attention implementation
│   │   ├── multihead.py        # Multi-head attention
│   │   └── cross.py            # Cross-attention implementation
│   ├── encoder/                # Encoder components
│   │   ├── __init__.py
│   │   ├── layer.py            # Single encoder layer
│   │   └── model.py            # Complete encoder model
│   ├── decoder/                # Decoder components
│   │   ├── __init__.py
│   │   ├── layer.py            # Single decoder layer
│   │   └── model.py            # Complete decoder model
│   ├── embedding.py            # Token embeddings
│   ├── positional_encoding.py  # Positional encoding
│   ├── feed_forward.py         # Feed-forward network
│   └── residual_add_norm.py    # Residual connections & layer norm
├── data/                       # Dataset directory
│   ├── .gitkeep               # Preserves directory structure
│   └── TinyStories.txt        # Training dataset (not in git)
├── experiments/               # Jupyter notebooks
│   └── Transformers.ipynb    # Experimental notebook
├── config.py                 # Configuration management
├── config.toml              # Model and training configuration
├── dataset.py               # Dataset loading and preprocessing
├── tokenizer.py             # Tokenization utilities
├── training.py              # Training loop implementation
├── model.py                 # Model creation and orchestration
├── main.py                  # CLI entry point
├── log_config.py            # Logging configuration
├── pyproject.toml           # Project metadata and dependencies
└── README.md
```

## 🏗️ Architecture Details

### Multi-Head Self-Attention

- **8 attention heads** with 64 dimensions each (512 total)
- **Scaled dot-product attention** with optional causal masking
- **Separate Q, K, V projections** for each head
- **Output projection** to combine all heads

### Encoder-Decoder Architecture

- **Encoder**: Bidirectional self-attention for input processing
- **Decoder**: Masked self-attention + cross-attention for autoregressive generation
- **Cross-attention**: Allows decoder to attend to encoder outputs

### Feed-Forward Network

- **Two linear layers**: 512 → 2048 → 512
- **ReLU activation** between layers
- **Dropout regularization** after activation

### Residual Connections & Normalization

- **Post-norm architecture**: LayerNorm applied after residual addition
- **Residual connections** around attention and feed-forward blocks
- **Separate normalization** for each sub-layer

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
from transformer.attentions.multihead import MultiHeadAttention
from transformer.attentions.self import SelfAttention

# Encoder attention (bidirectional)
encoder_attention = MultiHeadAttention(
    attention_type=SelfAttention,
    n_heads=8,
    hidden_size=512,
    max_seq_len=1024,
    dropout_pe=0.1,
    masking=False  # No causal masking
)

# Decoder attention (causal)
decoder_attention = MultiHeadAttention(
    attention_type=SelfAttention,
    n_heads=8,
    hidden_size=512,
    max_seq_len=1024,
    dropout_pe=0.1,
    masking=True  # Causal masking enabled
)
```

### Adjusting Model Size

```python
from config import Config

# Smaller model configuration
small_config = {
    "hidden_size": 256,
    "n_heads": 4,
    "ff_hidden_size": 1024,
    "n_layers": 4
}

# Larger model configuration
large_config = {
    "hidden_size": 1024,
    "n_heads": 16,
    "ff_hidden_size": 4096,
    "n_layers": 12
}
```

## 📊 Training

The implementation includes:

- **TinyStories dataset** support for quick experimentation
- **Configurable training loop** with encoder-decoder coordination
- **Comprehensive logging** with configurable log levels
- **Configuration management** with TOML-based settings
- **Flexible tokenization** with multiple tokenizer backends

### Training Pipeline

1. **Data Loading**: TinyStories dataset with configurable preprocessing
2. **Tokenization**: Support for tiktoken and custom tokenizers
3. **Model Creation**: Automatic encoder-decoder instantiation
4. **Training Loop**: Batch processing with proper encoder-decoder flow

## 🔬 Current Status

- ✅ **Encoder**: Fully implemented and tested
- ✅ **Decoder**: Complete architecture with cross-attention
- ✅ **Cross-Attention**: Fully implemented and functional
- ⚠️ **Training Loop**: Forward pass implemented, needs loss computation and backpropagation
- ✅ **Configuration System**: TOML-based configuration management
- ✅ **Dataset Integration**: TinyStories dataset with tokenization support

## 🤝 Contributing

This is an educational implementation. Contributions that improve clarity, add documentation, or enhance the learning experience are welcome!

## 📚 References

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation guide

## 📄 License

MIT License - Feel free to use this code for educational purposes.

---

**Note**: This implementation includes both encoder and decoder components of the Transformer architecture with fully functional cross-attention. The core architecture is complete and ready for sequence-to-sequence tasks. The main remaining work is completing the training loop with proper loss computation and optimization.
