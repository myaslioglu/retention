import torch
import torch.nn as nn
from config import Config
from arch.encoder.layer import EncoderLayer
import logging
from arch.embedding import Embeddings
from arch.positional_encoding import PositionalEncoding

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
        Represents a transformer-based encoder model, comprising token embeddings,
        positional encodings, and multiple encoder layers.

        :param vocab_size: The size of the vocabulary used in the token embedding layer.
        :param hidden_size: The dimensionality of token and positional embeddings as well as hidden states.
        :param seq_len: The maximum sequence length processed by the encoder.
        :param dropout_pe: Dropout probability applied to positional encoding.
        :param n_layers: The number of transformer encoder layers in the model.
        :param n_heads: The number of attention heads in each encoder layer.
        :param ff_size: The hidden layer dimensionality of the feed-forward network in each encoder layer.
        :param d_k: Dimensionality of each attention head.
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

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        inp_embd = self.token_embedding(x)
        out = self.position_encoding(inp_embd)
        for layer_id, enc_layer in enumerate(self.encoder_layers, start=1):
            logger.debug(f"Encoder Layer: {layer_id}")
            out = enc_layer(out, pad_mask)
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
    max_seq_len: int = conf.model.max_seq_len
    vocab_size: int = conf.model.vocab_size
    dropout_pe: float = conf.model.dropout_pe
    n_heads: int = conf.model.n_heads
    d_k: int = conf.model.get("d_k", None)
    ff_hidden_size: int = conf.model.ff_hidden_size
    n_layers: int = conf.model.n_layers

    return Encoder(vocab_size, hidden_size, max_seq_len, dropout_pe,
                   n_layers, n_heads, ff_hidden_size, d_k)
