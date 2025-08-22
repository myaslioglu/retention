import logging
import torch
from model import TransformerModel
from utils import BatchTensors

logger = logging.getLogger(__name__)

# def train_with_BLEU(model: Model, dataset, batch_size: int, epoch: int):
#     pass

def train_batch_CE(model: TransformerModel, batch: BatchTensors, loss_fn) -> torch.Tensor:
    """
    Trains one step in an encoder-decoder model by passing the source and target tokens
    through the complete model forward pass.

    :param model: The encoder-decoder model consists of an encoder, decoder,
        and classifier.
    :type model: Model
    :param batch: The batch containing src_batch_X, tgt_batch_X, and tgt_batch_y tensors
    :type batch: NamedTuple
    :return: The model's output logits after processing the input sequences.
    :rtype: torch.Tensor

    """
    src_batch_X = batch.src_batch_X
    tgt_batch_X = batch.tgt_batch_X
    tgt_batch_y = batch.tgt_batch_y
    src_batch_X_pad_mask = batch.src_batch_X_pad_mask

    logits = model.forward(src_batch_X, tgt_batch_X, src_batch_X_pad_mask)  # [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
    
    # Calculate loss if a loss function is provided
    # CrossEntropy Loss expects the logits for all the inputs in the batch of size [BATCH_SIZE, VOCAB_SIZE, SEQ_LEN]
    # So we transpose the logits
    if loss_fn is not None:
        return loss_fn(logits.transpose(1, 2), tgt_batch_y)  # [B, S, V] -> [B, V, S]
    
    # Return infinite loss
    return torch.tensor(float('inf'), device=logits.device)
