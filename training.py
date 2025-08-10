import logging
import torch

logger = logging.getLogger(__name__)

def train(encoder_model, decoder_model, dataset, batch_size: int, epoch: int):
    """
    Trains the provided encoder and decoder models using the given dataset, processing it in batches
    of the specified size. The function accumulates data from the dataset in a buffer until it matches 
    the batch size, then processes the batch by invoking `train_one()`. Training proceeds through the 
    dataset until the specified number of batches is completed.

    :param encoder_model: The encoder model to be trained.
    :type encoder_model: nn.Module
    :param decoder_model: The decoder model to be trained.
    :type decoder_model: nn.Module
    :param dataset: Iterable dataset used for training.
    :type dataset: Iterable
    :param batch_size: Number of samples per batch.
    :type batch_size: int
    :param epoch: Number of batches to process (note: this is number of batches, not epochs).
    :type epoch: int
    :return: None.
    """
    count = 0
    buffer = []
    for data in dataset:
        if count == epoch:
            break
        buffer.append(data)
        if len(buffer) == batch_size:
            count += 1
            logger.info(f"Processing batch {count}")
            train_one(encoder_model, decoder_model, torch.stack(buffer))
            buffer = []

def train_one(encoder_model, decoder_model, x: torch.Tensor):
    """
    Processes a single batch using the provided encoder and decoder models. Logs the
    input and output shapes during the process. Note: This function currently only
    performs forward pass without actual training (no loss computation or backpropagation).

    :param encoder_model: The encoder model to process the input.
    :type encoder_model: nn.Module
    :param decoder_model: The decoder model to process the encoder output.
    :type decoder_model: nn.Module
    :param x: Input tensor representing a batch of data.
    :type x: torch.Tensor
    :return: None (currently doesn't return anything).
    """
    logger.info(f"Training on batch: {x.shape}")
    encoder_output = encoder_model(x)
    output = decoder_model(x, encoder_output)
    logger.info(f"Training output: {output.shape}")