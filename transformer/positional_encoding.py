"""
Transformers can preserve the context but can't preserve the order
+---+----------------------------------------------+--------------------------------------------------+
|   | RNN                                          | Transformer                                      |
+---+----------------------------------------------+--------------------------------------------------+
| 1 | Preserves context and order.                 | Preserves context  but cannot natively           |
|   |                                              | preserve order.                                  |
+---+----------------------------------------------+--------------------------------------------------+
| 2 | Processes one token per timestep, passing    | Uses self-attention: many tokens are handled     |
|   | hidden state forward → slow & costly on      | in parallel (with GPU support), so long          |
|   | long sequences.                              | contexts train faster and cheaper.               |
+---+----------------------------------------------+--------------------------------------------------+

# Mayukh kills Lion
# Lion kills Mayukh

In the above 2 sentences, attention for kills will be the same in transformer,
but they mean entirely opposite just because of positioning.

++++++++++++ THIS MODULE IS USED TO ADD THAT POSITIONING DETAIL TO A TRANSFORMER WHICH WAS MISSING ++++++++++++
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Encodes the positional information of input tensors for transformers.

    This class implements a positional encoding mechanism that is commonly used
    in transformer-based neural network architectures. Positional encoding is
    used to provide information about the relative or absolute position of tokens
    in a sequence. This encoding is added to the token embeddings and enables
    transformers to take the order of tokens into account, which is necessary
    since transformers do not have a built-in sense of sequence order.
    """
    def __init__(self, seq_len: int, hidden_size: int, dropout: float):
        """
        Represents a class that precomputes positional encodings for a fixed sequence length and hidden size,
        used in transformer-based models. Positional encodings inject information about the position of
        tokens in the sequence into the model. This implementation uses sine and cosine functions
        to compute the encodings and stores them in a buffer, making them available for the forward pass.

        :param seq_len: The length of the sequence for which positional encodings are calculated.
        :param hidden_size: The size of the hidden representation (or embedding dimension) used in the model.
        :param dropout: Dropout rate to apply to the positional encodings.
        """
        super().__init__()
        self.d_model = hidden_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # We can have at-most seq_len number of positions for the input
        # because the size of the input can't be more than seq_len
        pos_encodings = torch.zeros(self.seq_len, self.d_model, dtype=torch.float32)
        log_factor = math.log(10000.0)
        exponent_constant = -torch.div(log_factor, self.d_model)
        exponent = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * exponent_constant
        )
        # Now fill in the even indices with sin
        pos = torch.arange(0, self.seq_len, dtype=torch.float32).reshape(self.seq_len, 1) # Make it column-vector
        pos_encodings[:, 0::2] = torch.sin(pos * exponent)
        # And odd indices with cos
        pos_encodings[:, 1::2] = torch.cos(pos * exponent)

        # Add a batch dimension with value 1
        pos_encodings = pos_encodings.reshape(1, self.seq_len, self.d_model)

        self.register_buffer('pe', pos_encodings)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :] # Take all the batch, only up to seq len and for all the columns (hidden_size)
        return self.dropout(x) # [BATCH, SEQ_LEN, HIDDEN_SIZE]




