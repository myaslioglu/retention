import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Handles the conversion of input tokens into embeddings, providing scaled
    embeddings for further processing. The class is designed to initialize
    an embedding layer based on the vocabulary size and hidden dimension.

    Embedding layers works just like a lookup table where all the tokens
    from the vocabulary are converted into a `hidden_size` vector

    :ivar d_model: Represents the dimension of the embeddings.
    :type d_model: Int
    :ivar n_vocab: Represents the size of the vocabulary.
    :type n_vocab: Int
    :ivar embedding: Embedding layer that maps token indices to embeddings.
    :type embedding: Nn.Embedding
    :ivar factor: a Scaling factor for embeddings, derived as the square root
        of the embedding dimension.
    :type factor: Torch.Tensor
    """
    def __init__(self, vocab_size: int, hidden_size: int):
        """Initialize the input embeddings layer.
        Args:
            vocab_size (int): Size of the vocabulary
            hidden_size (int): Dimension of the embeddings
        """
        super().__init__()
        self.d_model = torch.tensor(hidden_size)
        self.n_vocab = vocab_size
        self.embedding = nn.Embedding(self.n_vocab, hidden_size)
        self.factor = torch.sqrt(self.d_model)

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
