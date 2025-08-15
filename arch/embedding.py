import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    Represents an embeddings layer for input tokens in a neural network.

    This class provides functionality to map input token indices into
    dense vector representations (embeddings) and apply appropriate
    scaling as described in relevant literature.

    :ivar embedding: Reference to an embedding layer of the model.
    :type embedding: torch.nn.Embedding
    :ivar factor: Scaling factor applied to the embeddings.
    :type factor: float
    """
    def __init__(self, vocab_size: int, hidden_size: int):
        """Initialize the input embeddings layer.
        Args:
            vocab_size (int): Size of the vocabulary
            hidden_size (int): Dimension of the embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.factor = math.sqrt(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Convert input tokens into embeddings.
        Args:
            x (torch.Tensor): Input tensor of token indices

        Returns:
            torch.Tensor: Scaled embeddings
        """
        # The paper https://arxiv.org/pdf/1706.03762 at section 3.4
        # Scales the embedding layer output with the `factor`
        return self.embedding(x) * self.factor # [BATCH, SEQ_LEN, HIDDEN_SIZE]
