"""
Implements multi-head attention mechanism.

This module contains the implementation of the multi-head attention mechanism
used in neural network architectures like Transformers. It enables the model
to focus on different parts of the input sequence with multiple attention heads,
capturing richer contextual relationships and improving model performance.
"""
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Implements a Self-Attention mechanism for use in neural network architectures.

    This class defines a self-attention mechanism in which each element in the input
    sequence attends to all other elements, creating a representation of the input
    that considers relationships between sequence elements. It employs separate
    learnable weight matrices for query, key, and value computations.

    :ivar d_model: Dimensionality of the input feature space (hidden size).
    :type d_model: Int
    :ivar d_k: Dimensionality of the key, value, and query vectors.
    :type d_k: Int-    :ivar W_q: Learnable linear transformation for computing query vectors.
    :type W_q: Nn.Linear
    :ivar W_k: Learnable linear transformation for computing key vectors.
    :type W_k: Nn.Linear
    :ivar W_v: Learnable linear transformation for computing value vectors.
    :type W_v: Nn.Linear
    """

    def __init__(self, hidden_size: int, max_seq_len: int,
                 d_k: int, dropout_pe: float, masking: bool = False):
        """
        Initializes the instance of the class with the specified parameters. This
        includes the creation of linear layers for key, value, and query matrices
        used in attention mechanisms, as well as a dropout layer.

        :param hidden_size: The size of the input hidden feature dimension.
        :param d_k: Dimensionality of the key, value, and query representations.
        :param dropout_pe: The dropout probability to be applied to the attention
            mechanism outputs to help regularize the model.
        """
        super().__init__()
        self.d_model = hidden_size
        self.d_k = d_k

        # Create 3 matrices for Key, Value, Query
        self.W_q = nn.Linear(hidden_size, d_k)  # Query matrix
        self.W_k = nn.Linear(hidden_size, d_k)  # Key matrix
        self.W_v = nn.Linear(hidden_size, d_k)  # Value matrix

        # -1 represents the last index of [BATCH, SEQ_LEN, SEQ_LEN]
        self.softmax = nn.Softmax(dim=-1) # Used in attention scores

        self.attn_dropout = nn.Dropout(p=dropout_pe)

        # Create the masking buffer
        self.masking = masking
        inf: torch.Tensor = torch.full((max_seq_len, max_seq_len), float("-inf"))
        self.register_buffer('causal_mask', torch.triu(inf, diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [BATCH, SEQ_LEN, d_k]
        Q = self.W_q(x) # Query Vector
        K = self.W_k(x) # Key Vector
        V = self.W_v(x) # Value Vector

        # Perform Q.K
        # Q -> [BATCH, SEQ_LEN, d_k]
        # K -> [BATCH, SEQ_LEN, d_k]
        # So we need to transpose K to calculate dot product
        K_t = SelfAttention.transpose(K)
        scores = Q @ K_t

        # Now scale the attention scores
        scores = scores / math.sqrt(self.d_k)

        # Apply masking
        if self.masking:
            _, SeqLen, _ = scores.shape
            mask: torch.Tensor = self.causal_mask[:SeqLen, :SeqLen]
            scores = scores + mask

        # Apply Softmax
        scores = self.softmax(scores) # [BATCH, SEQ_LEN, SEQ_LEN]

        # Apply dropout
        W = self.attn_dropout(scores)

        # Linear Projection
        return W @ V # [BATCH, SEQ_LEN, d_k]

    @staticmethod
    def transpose(key: torch.Tensor) -> torch.Tensor:
        """
        Transposes the input tensor along specific dimensions.

        This function takes a tensor and swaps its second and third dimensions.
        It is useful for preparing the key tensor in various machine learning
        contexts, such as attention mechanisms in neural networks.

        :param key: A tensor to be transposed.
        :type key: Torch.Tensor
        :return: A tensor with its second and third dimensions swapped.
        :rtype: Torch.Tensor
        """

        # Take the key vector of dim [BATCH, SEQ_LEN, d_k]
        # Swap dim 1 and dim 2
        return key.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    """
    Implements a multi-head self-attention mechanism. This allows the model to attend
    to different parts of the input sequence independently using separate attention heads.
    The outputs of all attention heads are concatenated and projected through a linear
    transformation.

    :ivar d_k: The size of the query/key vectors used in each attention head. If not
        provided during initialization, it will be calculated as `hidden_size // n_heads`.
    :type d_k: Int
    :ivar self_attention_heads: A list of self-attention heads, one for each attention head.
    :type self_attention_heads: Nn.ModuleList
    :ivar W_o: The linear projection layer to combine and project the concatenated
        outputs of all attention heads back to the `hidden_size` dimension.
    :type W_o: Nn.Linear
    """

    def __init__(self, hidden_size: int, max_seq_len: int, n_heads: int,
                 dropout_pe: float, masking: bool, d_k: int | None = None):
        """
        Initializes the class instance and configures the multi-head self-attention mechanism.
        Validates input parameters for compatibility, including ensuring the key dimension is not
        greater than the hidden size and that the number of heads divides the hidden size evenly.

        :param hidden_size: The size of the input representation or embeddings.
        :param n_heads: The number of attention heads.
        :param d_k: (Optional) The size of the query/key vectors. If not provided, it will be
            automatically calculated as `hidden_size // n_heads`.
        :raises ValueError: If the provided key dimension exceeds the hidden size.
        :raises ValueError: If the number of attention heads does not evenly divide the hidden size.
        """

        super().__init__()
        if d_k and d_k > hidden_size:
            raise ValueError("Key dimension must be less than or equal to hidden size")
        else:
            if hidden_size % n_heads != 0:
                raise ValueError("Number of heads must divide hidden size evenly")
            d_k = hidden_size // n_heads

        self.d_k = d_k
        # Create `n_heads` number of self-attention heads
        self.self_attention_heads = nn.ModuleList([
            SelfAttention(hidden_size=hidden_size, max_seq_len=max_seq_len,
                          d_k=self.d_k, dropout_pe=dropout_pe, masking=masking)
            for _ in range(n_heads)
        ])

        # Create the W_o for Linear projection
        self.W_o = nn.Linear(d_k * n_heads, hidden_size)
        self.out_dropout = nn.Dropout(p=dropout_pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = []
        # Pass the inputs with all the attn heads
        for attn in self.self_attention_heads:
            head_outputs.append(attn(x))

        # Concat all the head's output horizontally
        # dim = -1 indicate the last dimension wise which is hidden_size
        concat_outputs: torch.Tensor = torch.cat(head_outputs, dim=-1) # [BATCH_SIZE, SEQ_LEN, d_k]

        # Project the output with Linear layer
        # This ensures
        # # 1. Correct output dim [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        # # 2. Learns the combination of head outputs
        # # 3. Removes any redundant head's output
        projected_output: torch.Tensor = self.W_o(concat_outputs)

        # Apply the dropout
        return self.out_dropout(projected_output)
