import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Implements a feed-forward neural network module commonly used in arch
    architectures.

    This module includes two linear layers with a non-linear ReLU activation
    function in between and a Dropout layer for regularization. It is typically
    applied to model intermediate transformations within arch blocks.

    :ivar W_1: Linear transformation layer mapping the input to the intermediate
       hidden layer.
    :type W_1: nn.Linear
    :ivar W_2: Linear transformation layer mapping the intermediate hidden
       representation back to the original feature size.
    :type W_2: nn.Linear
    :ivar relu: ReLU activation function applied after the first linear transformation.
    :type relu: nn.ReLU
    :ivar dropout: Dropout layer used to regularize the output from the feed-forward
       network.
    :type dropout: nn.Dropout
    """
    def __init__(self, hidden_size: int,
                 ff_hidden_size: int,
                 dropout_pe: float):
        """
        Initializes the feed-forward neural network module typically used in arch
        architectures. This module consists of two linear transformations with a ReLU
        activation function in between, followed by a Dropout layer for regularization.

        :param hidden_size: The size of the input and output features for the feed-forward
           network.
        :param ff_hidden_size: The size of the intermediate hidden layer in the feed-forward
           network.
        :param dropout_pe: The dropout probability applied to the output of the non-linear
           activation.
        """
        super().__init__()
        self.W_1 = nn.Linear(hidden_size, ff_hidden_size)
        self.W_2 = nn.Linear(ff_hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.W_2(x)
