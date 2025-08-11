import torch.nn as nn
import torch
import math


class CrossAttention(nn.Module):

    def __init__(self, hidden_size: int, max_seq_len: int,
                 d_k: int, dropout_pe: float, masking: bool = False):
        super().__init__()
        self.d_model = hidden_size
        self.d_k = d_k

        # Create 3 matrices for Key, Value, Query
        self.W_q = nn.Linear(hidden_size, d_k)  # Query matrix
        self.W_k = nn.Linear(hidden_size, d_k)  # Key matrix
        self.W_v = nn.Linear(hidden_size, d_k)  # Value matrix

        # -1 represents the last index of [BATCH, SEQ_LEN, SEQ_LEN]
        self.softmax = nn.Softmax(dim=-1)  # Used in attention scores

        self.attn_dropout = nn.Dropout(p=dropout_pe)

        # Masking is not required in cross attention
        _ = masking

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # // NOTE: We pass encoder output to the value and key vector
        Q = self.W_q(x)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)

        # Perform Q.K
        # Q -> [BATCH, SEQ_LEN, d_k]
        # K -> [BATCH, SEQ_LEN, d_k]
        # So we need to transpose K to calculate dot product
        K_t = CrossAttention.transpose(K)
        scores = Q @ K_t

        # Now scale the attention scores
        scores = scores / math.sqrt(self.d_k)

        # Apply Softmax
        scores = self.softmax(scores)  # [BATCH, SEQ_LEN, SEQ_LEN]

        # Apply dropout
        W = self.attn_dropout(scores)

        # Linear Projection
        return W @ V

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