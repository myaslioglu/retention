import torch.nn as nn
import torch

from transformer.attention import MultiHeadAttention
from transformer.residual_add_norm import ResidualAddNorm
from transformer.feed_forward import FeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, dropout_pe: float,
                 n_heads: int, ff_hidden_size: int,
                 d_k: int|None = None):
        """
        Initializes the transformer layer with token embeddings, positional
        encoding, multi-head attention mechanism, residual connections,
        and a feed-forward network.

        :param vocab_size: Size of the vocabulary for embedding initialization.
        :param hidden_size: Dimensionality of the token embeddings and model hidden states.
        :param seq_len: Sequence length for positional encoding.
        :param dropout_pe: Dropout rate for positional encoding, multi-head attention,
                           and feed-forward layers.
        :param n_heads: Number of attention heads in the multi-head attention mechanism.
        :param ff_hidden_size: Dimensionality of the hidden layer in the feed-forward network.
        :param d_k: Dimensionality of the key (and query) vectors in attention. If None,
                    a default dimension is computed using hidden size and number of heads.
        """
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(hidden_size, max_seq_len,
                                                  n_heads, dropout_pe, False, d_k)
        self.residual_add_norm_attn = ResidualAddNorm(n_features=hidden_size)
        self.residual_add_norm_ff = ResidualAddNorm(n_features=hidden_size)
        self.feed_forward = FeedForward(hidden_size,
                                        ff_hidden_size=ff_hidden_size,
                                        dropout_pe=dropout_pe)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        attn_output = self.multi_head_attn(token_ids)
        res_norm = self.residual_add_norm_attn(token_ids, attn_output)
        ff_output = self.feed_forward(res_norm)
        return self.residual_add_norm_ff(res_norm, ff_output)

