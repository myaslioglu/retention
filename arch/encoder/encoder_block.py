# arch/encoder/encoder_block.py
from torch import nn
from ..residual_add_norm import ResidualAddNorm
from ..feed_forward import PositionwiseFeedForward
from ..attentions import make_attention

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden_size, dropout, max_seq_len, attention_kind="mha"):
        super().__init__()
        self.attn = make_attention(
            kind=attention_kind,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.attn_block = ResidualAddNorm(d_model, self.attn, dropout)
        self.ff = PositionwiseFeedForward(d_model, ff_hidden_size, dropout)
        self.ff_block = ResidualAddNorm(d_model, self.ff, dropout)

    def forward(self, x, src_pad_mask=None):
        x = self.attn_block(x, pad_mask=src_pad_mask, is_causal=True)
        x = self.ff_block(x)
        return x
