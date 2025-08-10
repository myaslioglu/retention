import logging
import torch

logger = logging.getLogger(__name__)

def train(encoder_model, decoder_model, dataset, batch_size: int, epoch: int):
    """
    Trains the provided model using the given dataset, processing it in batches
    of the specified size for the defined number of epochs. The function
    accumulates data from the dataset in a buffer until it matches the batch
    size, then processes the batch by invoking `train_one()`. Training proceeds
    through the dataset until the specified number of epochs is completed.

    :param model: Model to be trained.
    :type model: Any
    :param dataset: Iterable dataset used for training.
    :type dataset: Iterable
    :param batch_size: Number of samples per batch.
    :param epoch: Number of epochs for training.
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
    Trains a single batch using the provided model and input tensor. Logs the
    input and output shapes during the process.

    :param model: The model to be trained.
    :type model: typing.Any
    :param x: Input tensor representing a batch of data.
    :return: The output tensor resulting from the model's forward pass.
    :rtype: torch.Tensor
    """
    logger.info(f"Training on batch: {x.shape}")
    encoder_output = encoder_model(x)
    output = decoder_model(x, encoder_output)
    logger.info(f"Training output: {output.shape}")