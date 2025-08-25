import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from config import Config
import logging
from collections import namedtuple

# Define BatchTensors here to avoid circular imports
BatchTensors = namedtuple('BatchTensors', ['src_batch_X', 'tgt_batch_X', 
                                           'tgt_batch_y', 'src_batch_X_pad_mask', 'tgt_batch_X_pad_mask'])


logger = logging.getLogger(__name__)

def collate_fn(batch, pad_id: int, bos_id: int, eos_id: int, max_seq_len: int):
    """
    Collate function for batching tokenized sequence pairs for transformer training.
    This function processes a batch of (source, target) token sequences and prepares them
    for transformer model training by adding special tokens and padding to a fixed length.
    Args:
        batch: List of tuples containing (src_tokens, tgt_tokens) pairs where each
               element is a list of token IDs.
        pad_id (int): Token ID used for padding sequences to max_seq_len.
        bos_id (int): Beginning-of-sequence token ID added to decoder input.
        eos_id (int): End-of-sequence token ID added to encoder input and decoder target.
        max_seq_len (int): Maximum sequence length for padding/truncation.
    Returns:
        BatchTensors: Named tuple containing:
            - src_batch_X: Tensor of shape (batch_size, max_seq_len) for encoder input
              Format: [SRC_TOKENS + EOS + PAD...]
            - tgt_batch_X: Tensor of shape (batch_size, max_seq_len) for decoder input
              Format: [BOS + TGT_TOKENS + PAD...]
            - tgt_batch_y: Tensor of shape (batch_size, max_seq_len) for decoder target
              Format: [TGT_TOKENS + EOS + PAD...]
    Note:
        - Sequences longer than max_seq_len-1 are truncated to accommodate special tokens
        - All tensors are created on CPU with dtype=torch.long
        - The function assumes BatchTensors is a named tuple or similar container class
    """
    
    # Create tensors on CPU first
    src_batch_X = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
    tgt_batch_X = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
    tgt_batch_y = torch.zeros(len(batch), max_seq_len, dtype=torch.long)

    for i, (src_tkn, tgt_tkn) in enumerate(batch):
        # For encoder input: SRC_TOKENS + EOS + [PAD]
        src_X = src_tkn[:max_seq_len - 1] + [eos_id]

        # For decoder input: BOS + TGT_TOKENS + [PAD]
        d_tkn = tgt_tkn[:max_seq_len - 1]
        tgt_X = [bos_id] + d_tkn

        # For decoder target: TGT_TOKENS + EOS + [PAD]
        tgt_y = d_tkn + [eos_id]

        # Pad sequences
        src_X = src_X + [pad_id] * (max_seq_len - len(src_X))
        tgt_X = tgt_X + [pad_id] * (max_seq_len - len(tgt_X))
        tgt_y = tgt_y + [pad_id] * (max_seq_len - len(tgt_y))

        # Create tensors on CPU
        src_batch_X[i] = torch.tensor(src_X, dtype=torch.long)
        tgt_batch_X[i] = torch.tensor(tgt_X, dtype=torch.long)
        tgt_batch_y[i] = torch.tensor(tgt_y, dtype=torch.long)

    src_batch_X_pad_mask = (src_batch_X == pad_id)
    tgt_batch_X_pad_mask = (tgt_batch_X == pad_id)
    return BatchTensors(src_batch_X, tgt_batch_X, tgt_batch_y, 
                        src_batch_X_pad_mask, tgt_batch_X_pad_mask)

def get_dataloader(ds: Dataset, config: Config):
    """
    Fixed dataloader that doesn't pass device to collate function.
    Device transfer will happen in the model.forward() method.
    Enables pin_memory for GPU performance when CUDA is available.
    Optimized for memory efficiency.
    """
    # Determine if we should use pin_memory for GPU optimization
    use_pin_memory = torch.cuda.is_available()
    
    return DataLoader(
        dataset=ds.dataset,
        batch_size=config.training.batch_size,
        collate_fn=lambda _batch: collate_fn(
            _batch,
            pad_id=ds.tokenizer.pad_id,
            bos_id=ds.tokenizer.bos_id,
            eos_id=ds.tokenizer.eos_id,
            max_seq_len=config.model.max_seq_len
        ),
        num_workers=0,  # Avoid multiprocessing issues - reduces memory overhead
        pin_memory=use_pin_memory,  # Enable for GPU performance when CUDA available
        drop_last=True,  # Drop last incomplete batch to maintain consistent batch sizes
    )


def get_device(config: Config) -> torch.device:
    """
    Determines the device to use for training based on configuration and availability.
    Supports CUDA (NVIDIA) and CPU.
    
    :param config: Configuration object containing device preference
    :type config: Config
    :return: The device to use for training
    :rtype: torch.device
    """
    device_preference = getattr(config.training, 'device', 'auto').lower()
    
    if device_preference == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info("No GPU acceleration available, auto-selected CPU")
    elif device_preference == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using configured CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
    elif device_preference == 'cpu':
        device = torch.device('cpu')
        logger.info("Using configured CPU device")
    else:
        logger.warning(f"Unknown device preference '{device_preference}'")
        raise ValueError(f"Invalid device configuration: '{device_preference}'. Must be 'auto', 'cuda', or 'cpu'")
    
    return device
