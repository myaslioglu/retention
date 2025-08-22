import logging
import torch
from torch.utils.data import DataLoader

from model import TransformerModel
from utils import BatchTensors

logger = logging.getLogger(__name__)

# def train_with_BLEU(model: Model, dataset, batch_size: int, epoch: int):
#     pass

def train_epoch_avg_CE(model: TransformerModel,
                       train_data_loader: DataLoader[BatchTensors],
                       loss_fn,
                       wandb_run=None) -> torch.Tensor:
    """
    Computes the average cross-entropy loss per batch for a given epoch during training.

    The function iterates through the provided training data loader, processes each batch by
    moving data to the device associated with the model, and calculates per-batch loss using
    the provided loss function. The computed batch losses are aggregated and averaged over
    the total number of batches to determine and return the mean loss for the epoch.

    :param wandb_run: The wandb run object to log to. If None, no logging will be done.
    :type wandb_run: wandb.Run
    :param model: The model to train. It should have a `device` attribute where computations
        should occur.
    :type model: TransformerModel
    :param train_data_loader: DataLoader object yielding batches of training data. Each batch is
        expected to provide tensors for source input, target input, target labels, and padding
        masks for respective inputs.
    :type train_data_loader: DataLoader[BatchTensors]
    :param loss_fn: A callable that computes the loss given model predictions and target values.
        The function should return a tensor containing the computed loss value.
    :type loss_fn: callable
    :return: A tensor representing the average loss per batch for the training epoch. If no
        batches exist in the loader, it returns `float('inf')` as the loss.
    :rtype: torch.Tensor
    """
    batch_loss = 0.0
    num_batches = 0
    
    for idx, batch in enumerate(train_data_loader):
        batch_on_device = BatchTensors(
            src_batch_X=batch.src_batch_X.to(model.device),
            tgt_batch_X=batch.tgt_batch_X.to(model.device),
            tgt_batch_y=batch.tgt_batch_y.to(model.device),
            src_batch_X_pad_mask=batch.src_batch_X_pad_mask.to(model.device),
            tgt_batch_X_pad_mask=batch.tgt_batch_X_pad_mask.to(model.device)
        )
        loss = train_batch_CE(model=model, batch=batch_on_device, loss_fn=loss_fn).item()
        logger.info(f"Batch: {idx} loss: {loss}")
        
        # Log to wandb if available
        if wandb_run is not None:
            wandb_run.log({"batch_loss": loss, "batch_idx": idx})
            
        batch_loss += loss
        num_batches += 1
    
    if num_batches == 0:
        return torch.tensor(float('inf'), device=model.device, dtype=torch.float32)
    # Return average loss per batch
    return torch.tensor(batch_loss / num_batches, device=model.device, dtype=torch.float32)

def train_batch_CE(model: TransformerModel, batch: BatchTensors, loss_fn) -> torch.Tensor:
    """
    Trains one step in an encoder-decoder model by passing the source and target tokens
    through the complete model forward pass.

    :param loss_fn:
    :param model: The encoder-decoder model consists of an encoder, decoder,
        and classifier.
    :type model: Model
    :param batch: The batch containing src_batch_X, tgt_batch_X, and tgt_batch_y tensors
    :type batch: NamedTuple
    :return: The model's output logits after processing the input sequences.
    :rtype: Torch.Tensor

    """
    src_batch_X = batch.src_batch_X
    tgt_batch_X = batch.tgt_batch_X
    tgt_batch_y = batch.tgt_batch_y
    src_batch_X_pad_mask = batch.src_batch_X_pad_mask
    tgt_batch_X_pad_mask = batch.tgt_batch_X_pad_mask

    logits = model.forward(src_batch_X, tgt_batch_X, 
                           src_batch_X_pad_mask, tgt_batch_X_pad_mask)  # [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
    
    # Calculate loss if a loss function is provided
    # CrossEntropy Loss expects the logits for all the inputs in the batch of size [BATCH_SIZE, VOCAB_SIZE, SEQ_LEN]
    # So we transpose the logits
    if loss_fn is not None:
        return loss_fn(logits.transpose(1, 2), tgt_batch_y)  # [B, S, V] -> [B, V, S]
    
    # Return infinite loss
    return torch.tensor(float('inf'), device=logits.device)
