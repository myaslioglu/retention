import logging
import torch
import torch.nn as nn
from config import Config
from transformer.embedding import Embeddings
from transformer.positional_encoding import PositionalEncoding
from transformer.decoder.layer import DecoderLayer

logger = logging.getLogger(__name__)
class Decoder(nn.Module):
    """
    """
    def __init__(self, vocab_size: int, hidden_size: int, seq_len: int, 
                 dropout_pe: float, n_layers:int, n_heads: int, ff_size: int, d_k:int):
        super().__init__()
        self.token_embedding = Embeddings(vocab_size=vocab_size, hidden_size=hidden_size)
        self.position_encoding = PositionalEncoding(seq_len, hidden_size, dropout_pe)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size=hidden_size,
                max_seq_len=seq_len,
                dropout_pe=dropout_pe,
                n_heads=n_heads,
                ff_hidden_size=ff_size,
                d_k=d_k) 
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        out_embd = self.token_embedding(x)
        out = self.position_encoding(out_embd)
        for dec_layer in self.decoder_layers:
            out = dec_layer(out, encoder_output)
        return out


def get_decoder(conf: Config) -> Decoder:
    hidden_size: int = conf.model.hidden_size
    seq_len: int = conf.model.seq_len
    vocab_size: int = conf.model.vocab_size
    dropout_pe: float = conf.model.dropout_pe
    n_heads: int = conf.model.n_heads
    d_k: int = conf.model.get("d_k", None)
    ff_hidden_size: int = conf.model.ff_hidden_size
    n_layers: int = conf.model.n_layers
    return Decoder(vocab_size, hidden_size, seq_len,
                   dropout_pe, n_layers, n_heads, ff_hidden_size, d_k)
