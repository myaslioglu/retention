import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from config import Config
import logging
from collections import namedtuple

# Define BatchTensors here to avoid circular imports
BatchTensors = namedtuple('BatchTensors', ['src_batch_X', 'tgt_batch_X', 'tgt_batch_y'])


logger = logging.getLogger(__name__)

def collate_fn(batch, pad_id: int, bos_id: int, eos_id: int, max_seq_len: int):
    """
    Fixed collate function that creates tensors on CPU first, then moves to device.
    This prevents memory issues during data loading.
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
    
    return BatchTensors(src_batch_X, tgt_batch_X, tgt_batch_y)

def get_dataloader(ds: Dataset, config: Config):
    """
    Fixed dataloader that doesn't pass device to collate function.
    Device transfer will happen in the model.forward() method.
    """
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
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False  # Not needed for CPU
    )


def get_device(config: Config) -> torch.device:
    """
    Determines the device to use for training based on configuration and availability.
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
    
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
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Auto-selected MPS device (Apple Silicon)")
            # Get system memory info for Apple Silicon
            try:
                import psutil
                memory_gb = psutil.virtual_memory().total / 1e9
                logger.info(f"System memory: {memory_gb:.1f} GB")
            except ImportError:
                logger.info("MPS device selected (install psutil for memory info)")
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
    elif device_preference == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using configured MPS device (Apple Silicon)")
            try:
                import psutil
                memory_gb = psutil.virtual_memory().total / 1e9
                logger.info(f"System memory: {memory_gb:.1f} GB")
            except ImportError:
                logger.info("MPS device configured (install psutil for memory info)")
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device('cpu')
    elif device_preference == 'cpu':
        device = torch.device('cpu')
        logger.info("Using configured CPU device")
    else:
        logger.warning(f"Unknown device preference '{device_preference}'")
        raise ValueError(f"Invalid device configuration: '{device_preference}'. Must be 'auto', 'cuda', 'mps', or 'cpu'")
    
    return device
