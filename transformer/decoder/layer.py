import torch
import torch.nn as nn
from transformer.attentions.multihead import MultiHeadAttention
from transformer.attentions.cross import CrossAttention
from transformer.residual_add_norm import ResidualAddNorm
from transformer.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self,hidden_size: int, max_seq_len: int, 
                 dropout_pe: float, n_heads: int, ff_hidden_size: int, d_k: int|None = None):
        super().__init__()

        # Mask = True for decoder layers
        self.masked_multi_head_attn = MultiHeadAttention(n_heads, hidden_size, max_seq_len, 
                                                         dropout_pe, True, d_k)
        self.residual_add_norm_ma = ResidualAddNorm(n_features=hidden_size)
        self.cross_multi_head_attn = CrossAttention()
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