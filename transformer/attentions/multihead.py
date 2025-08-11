import torch
import torch.nn as nn
from transformer.attentions.self import SelfAttention
from typing import Union

class MultiHeadAttention(nn.Module):
    """
    Implements a multi-head self-attention mechanism. This allows the model to attend
    to different parts of the input sequence independently using separate attention heads.
    The outputs of all attention heads are concatenated and projected through a linear
    transformation.

    :ivar d_k: The size of the query/key vectors used in each attention head. If not,
        provided during initialization, it will be calculated as `hidden_size // n_heads`.
    :type d_k: Int
    :ivar self_attention_heads: A list of self-attention heads, one for each attention head.
    :type self_attention_heads: Nn.ModuleList
    :ivar W_o: The linear projection layer to combine and project the concatenated
        outputs of all attention heads back to the `hidden_size` dimension.
    :type W_o: Nn.Linear
    """

    def __init__(self, attention_type,
                 n_heads: int, hidden_size: int, max_seq_len: int,
                 dropout_pe: float, masking: bool, d_k: Union[int, None] = None):
        """
        Initializes the class instance and configures the multi-head self-attention mechanism.
        Validates input parameters for compatibility, including ensuring the key dimension is not
        greater than the hidden size and that the number of heads divides the hidden size evenly.

        :param n_heads: The number of attention heads.
        :param hidden_size: The size of the input representation or embeddings.
        :param max_seq_len: Maximum sequence length for an attention mechanism.
        :param dropout_pe: Dropout probability for attention outputs.
        :param masking: Whether to apply causal masking (True for decoder, False for encoder).
        :param d_k: (Optional) The size of the query/key vectors. If not provided, it will be
            automatically calculated as `hidden_size // n_heads`.
        :raises ValueError: If the provided key dimension exceeds the hidden size.
        :raises ValueError: If the number of attention heads does not evenly divide the hidden size.
        """

        super().__init__()
        if d_k and d_k > hidden_size:
            raise ValueError("Key dimension must be less than or equal to hidden size")
        if hidden_size % n_heads != 0:
            raise ValueError("Number of heads must divide hidden size evenly")
        d_k = hidden_size // n_heads

        self.d_k = d_k
        self.IsSelfAttention = isinstance(attention_type, SelfAttention)
        # Create `n_heads` number of self-attention heads
        self.self_attention_heads = nn.ModuleList([
            attention_type(hidden_size=hidden_size, max_seq_len=max_seq_len,
                          d_k=self.d_k, dropout_pe=dropout_pe, masking=masking)
            for _ in range(n_heads)
        ])

        # Create the W_o for Linear projection
        self.W_o = nn.Linear(d_k * n_heads, hidden_size)
        self.out_dropout = nn.Dropout(p=dropout_pe)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        head_outputs = []

        if self.IsSelfAttention:
            # Pass the inputs with all the attn heads
            for attn in self.self_attention_heads:
                head_outputs.append(attn(x))
        else:
            # For Cross Attention, we give both input and encoder output
            for attn in self.self_attention_heads:
                head_outputs.append(attn(x, encoder_output))

        # Concat all the head's output horizontally
        # dim = -1 indicate the last dimension wise which is hidden_size
        concat_outputs: torch.Tensor = torch.cat(head_outputs, dim=-1) # [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]

        # Project the output with Linear layer
        # This ensures
        # # 1. Correct output dim [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        # # 2. Learns the combination of head outputs
        # # 3. Removes any redundant head's output
        projected_output: torch.Tensor = self.W_o(concat_outputs)

        # Apply the dropout
        return self.out_dropout(projected_output)
