import torch
import torch.nn as nn
from config import Config

class Classifier(nn.Module):
    """
    Summary of what the class does.

    This class represents a neural network model for classification tasks. It
    consists of a single fully connected layer followed by a softmax layer to
    compute probabilities. This model is designed to process input tensors
    and output probability distributions across the given vocabulary.

    :ivar linear: The fully connected linear layer that maps input features
        to output logits corresponding to the vocabulary size.
    :type linear: nn.Linear
    :ivar softmax: The softmax layer that converts logits to probability
        distributions.
    :type softmax: nn.Softmax
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor by applying a linear transformation followed by a softmax
        activation to generate probabilities.

        The `forward` method converts the input tensor into a probability distribution across
        the vocabulary dimensions by first applying a learned linear transformation, then using
        softmax for normalization.

        :param x: The input tensor of shape [BATCH, SEQ_LEN, FEATURE_SIZE].
        :type x: torch.Tensor
        :return: The output tensor of probabilities with shape [BATCH, SEQ_LEN, VOCAB_SIZE].
        :rtype: torch.Tensor
        """
        logits = self.linear(x) # [BATCH, SEQ_LEN, VOCAB_SIZE]
        # We don't need to apply softmax because we are using CrossEntropyLoss
        # prob = self.softmax(logits) # [BATCH, SEQ_LEN, VOCAB_SIZE]
        return logits


def get_classifier(conf: Config) -> Classifier:
    """
    Creates and returns a classifier instance based on the given configuration.

    The function extracts the `hidden_size` and `vocab_size` from the `conf`
    object and initializes a `Classifier` instance with these values. It is
    used to set up a classifier based on the model configuration.

    :param conf: The configuration object containing parameters for the model.
                 It must have `hidden_size` and `vocab_size` attributes accessible
                 under `conf.model`.
    :type conf: Config

    :return: A `Classifier` instance is initialized with the provided
             `hidden_size` and `vocab_size`.
    :rtype: Classifier
    """
    hidden_size: int = conf.model.hidden_size
    vocab_size: int = conf.model.vocab_size
    return Classifier(hidden_size, vocab_size)
