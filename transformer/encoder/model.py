import torch
import torch.nn as nn
from config import Config
from transformer.encoder.layer import EncoderLayer
import logging
from transformer.embedding import Embeddings
from transformer.positional_encoding import PositionalEncoding

logger = logging.getLogger(__name__)
class Encoder(nn.Module):
    """
    Represents a Transformer Encoder module which processes input sequences to generate
    contextualized embeddings. Commonly used in natural language processing tasks.

    This class is composed of token embeddings, positional encodings, and a stack of
    Transformer encoder layers. The Transformer Encoder allows sequence-based data to
    be processed efficiently using self-attention mechanisms.

    :ivar token_embedding: Embedding layer to convert input tokens into dense vector
        representations.
    :type token_embedding: Embeddings
    :ivar position_encoding: Positional encoding module to incorporate sequence
        position information into the embeddings.
    :type position_encoding: PositionalEncoding
    :ivar encoder_layers: List of Transformer encoder layers stacked sequentially
        for processing the input sequence.
    :type encoder_layers: nn.ModuleList
    """
    def __init__(self, vocab_size: int, hidden_size: int,
                 seq_len: int, dropout_pe: float,
                 n_layers: int, n_heads: int, ff_size: int, d_k:int):
        """
        Represents a transformer-based encoder module that includes token embeddings,
        positional encodings, and multiple layers of transformer encoder components.
        This module is designed to handle input sequences and generate contextualized
        representations leveraging attention mechanisms.

        :param vocab_size: Size of the input vocabulary.
        :param hidden_size: Dimensionality of the token embeddings and hidden state.
        :param seq_len: Maximum sequence length for input data.
        :param dropout_pe: Dropout probability for positional encoding.
        :param n_layers: Number of encoder layers.
        :param n_heads: Number of attention heads in each encoder layer.
        :param ff_size: Dimensionality of the feed-forward network hidden layer within the encoder.
        :param d_k: Dimensionality of the query and key vectors in the attention mechanism.
        """
        super().__init__()
        self.token_embedding = Embeddings(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(seq_len, hidden_size, dropout_pe)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_size=hidden_size,
                         max_seq_len=seq_len,
                         dropout_pe=dropout_pe,
                         n_heads=n_heads,
                         ff_hidden_size=ff_size,
                         d_k=d_k)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_embd = self.token_embedding(x)
        out = self.position_encoding(inp_embd)
        for enc_layer in self.encoder_layers:
            out = enc_layer(out)
        return out

def get_encoder(conf: Config) -> Encoder:
    """
    This function initializes and returns a TransformerEncoder instance based on the provided configuration and dataset.
    The encoder is configured using parameters including hidden size, vocabulary size, dropout rate, number of
    attention heads, feed-forward hidden size, optional dimensionality of the attention key,
    and number of encoding layers.

    :param conf: Configuration object containing model parameters such as hidden size, dropout probability,
        number of heads, feed-forward hidden size, and optionally the dimensionality of the attention key.
    :type conf: Config
    :return: An instance of TransformerEncoder initialized with parameters derived from the configuration and dataset.
    :rtype: Encoder
    """
    hidden_size: int = conf.model.hidden_size
    seq_len: int = conf.model.seq_len
    vocab_size: int = conf.model.vocab_size
    dropout_pe: float = conf.model.dropout_pe
    n_heads: int = conf.model.n_heads
    d_k: int = conf.model.get("d_k", None)
    ff_hidden_size: int = conf.model.ff_hidden_size
    n_layers: int = conf.model.n_layers

    return Encoder(vocab_size, hidden_size, seq_len, dropout_pe,
                   n_layers, n_heads, ff_hidden_size, d_k)


def get_decoder():
    pass
