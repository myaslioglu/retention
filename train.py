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
                       wandb_run=None,
                       max_batches: int = 100) -> torch.Tensor:
    """
    
    """
    batch_loss = 0.0
    num_batches = 0
    
    for idx, batch in enumerate(train_data_loader, start=1):
        # Limit number of batches to prevent memory overflow
        if idx >= max_batches:
            logger.info(f"Reached maximum batch limit of {max_batches}, stopping epoch")
            break
            
        batch_on_device = BatchTensors(
            src_batch_X=batch.src_batch_X.to(model.device),
            tgt_batch_X=batch.tgt_batch_X.to(model.device),
            tgt_batch_y=batch.tgt_batch_y.to(model.device),
            src_batch_X_pad_mask=batch.src_batch_X_pad_mask.to(model.device),
            tgt_batch_X_pad_mask=batch.tgt_batch_X_pad_mask.to(model.device)
        )
        
        try:
            loss = train_batch_CE(model=model, batch=batch_on_device, loss_fn=loss_fn).item()
            logger.info(f"Batch: {idx} loss: {loss: .5f}")

            # Wandb logging
            if wandb_run is not None:
                wandb_run.log({"batch_loss": loss, "batch_idx": idx})
                
            batch_loss += loss
            num_batches += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU out of memory at batch {idx}. Try reducing batch_size or model size.")
                # Clear cache and continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            else:
                raise e
        finally:
            # Clear cache after each batch to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
