import torch
import torch.nn as nn
from transformer.attentions.multihead import MultiHeadAttention
from transformer.attentions.cross import CrossAttention
from transformer.residual_add_norm import ResidualAddNorm
from transformer.feed_forward import FeedForward
from transformer.attentions.self import SelfAttention

class DecoderLayer(nn.Module):
    """
    Represents a single decoder layer for transformer-based models.

    This layer is designed to perform masked self-attention, cross-attention,
    and feed-forward transformations with residual connections and layer
    normalization applied to each stage. These components work together
    to facilitate effective learning and contextual understanding in the
    decoder part of the transformer architecture.

    :ivar masked_multi_head_attn: Layer for performing masked multi-head
        attention on the input sequence, ensuring future tokens remain masked.
    :type masked_multi_head_attn: MultiHeadAttention
    :ivar residual_add_norm_ma: Layer performing residual connection and
        normalization for the output of the masked multi-head attention.
    :type residual_add_norm_ma: ResidualAddNorm
    :ivar cross_multi_head_attn: Layer for performing cross-multi-head attention
        to incorporate information from the encoder's output.
    :type cross_multi_head_attn: MultiHeadAttention
    :ivar residual_add_norm_ca: Layer performing residual connection and
        normalization for the output of the cross-multi-head attention.
    :type residual_add_norm_ca: ResidualAddNorm
    :ivar feed_forward: Fully connected feed-forward layer applied after the
        attention mechanisms.
    :type feed_forward: FeedForward
    :ivar residual_add_norm_ff: Layer performing residual connection and
        normalization for the output of the feed-forward layer.
    :type residual_add_norm_ff: ResidualAddNorm
    """
    def __init__(self,hidden_size: int, max_seq_len: int,
                 dropout_pe: float, n_heads: int, ff_hidden_size: int, d_k: int|None = None):
        """
        Initializes a decoder layer consisting of masked multi-head attention, cross-multi-head
        attention, feed-forward network, and corresponding residual normalization components.

        The decoder layer uses the provided hyperparameters to define each sublayer:
        masked multi-head attention, which is commonly used in a decoder's
        self-attention mechanism, cross-multi-head attention, which integrates encoder
        information, and a feed-forward layer for additional processing. Each component is
        followed by residual connections and layer normalization.

        :param hidden_size: The size of the input and output features of each
            sublayer within the decoder.
        :param max_seq_len: The maximum sequence length the layer is designed to handle.
        :param dropout_pe: Dropout probability to be applied to internal operations
            such as attention computations and feed-forward layers.
        :param n_heads: The number of attention heads in the multi-head attention
            mechanism.
        :param ff_hidden_size: The hidden layer size for the feed-forward
            network.
        :param d_k: Optional. The dimensionality of each key/query vector in
            the attention mechanism. If not provided, it will default to a value
            based on the implementation.
        """
        super().__init__()

        # Mask = True for decoder masked multihead attention layer
        self.masked_multi_head_attn = MultiHeadAttention(SelfAttention, n_heads, hidden_size, max_seq_len,
                                                         dropout_pe, True, d_k)
        self.residual_add_norm_ma = ResidualAddNorm(n_features=hidden_size)

        # Mask = False for decoder cross-multihead attention layer
        self.cross_multi_head_attn = MultiHeadAttention(CrossAttention, n_heads, hidden_size, max_seq_len,
                                                        dropout_pe, False, d_k)
        self.residual_add_norm_ca = ResidualAddNorm(n_features=hidden_size)
        self.feed_forward = FeedForward(hidden_size,
                                        ff_hidden_size=ff_hidden_size,
                                        dropout_pe=dropout_pe)

        self.residual_add_norm_ff = ResidualAddNorm(n_features=hidden_size)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        masked_attn_output = self.masked_multi_head_attn(x)
        res_norm_ma = self.residual_add_norm_ma(x, masked_attn_output)
        cross_attn_output = self.cross_multi_head_attn(res_norm_ma, encoder_output)
        res_norm_ca = self.residual_add_norm_ca(res_norm_ma, cross_attn_output)
        ff_output = self.feed_forward(res_norm_ca)
        return self.residual_add_norm_ff(res_norm_ca, ff_output)
