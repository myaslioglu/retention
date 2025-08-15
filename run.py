from model import build_transformer
from config import Config
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from collections import namedtuple


def collate_fn(batch, pad_id: int,  bos_id: int, eos_id: int, max_seq_len: int, device: torch.device):
    src_batch_X = torch.zeros(len(batch), max_seq_len, dtype=torch.long, device=device)
    tgt_batch_X = torch.zeros(len(batch), max_seq_len, dtype=torch.long, device=device)
    tgt_batch_y = torch.zeros(len(batch), max_seq_len, dtype=torch.long, device=device)

    for i, (src_tkn, tgt_tkn) in enumerate(batch):
        #  For encoder input
        #  SRC_TOKENS + EOS + [PAD]
        src_X = src_tkn[:max_seq_len - 1] + [eos_id] # Make room for EOS

        # For decoder input
        # BOS + TGT_TOKENS + [PAD]
        d_tkn = tgt_tkn[:max_seq_len - 1]
        tgt_X = [bos_id] + d_tkn # Make room for BOS

        # For decoder target
        # TGT_TOKENS + EOS + [PAD]
        tgt_y = d_tkn + [eos_id] # Make room for EOS

        # Pad src_X and tgt_X
        src_X = src_X + [pad_id] * (max_seq_len - len(src_X))
        tgt_X = tgt_X + [pad_id] * (max_seq_len - len(tgt_X))
        tgt_y = tgt_y + [pad_id] * (max_seq_len - len(tgt_y))

        # Assign sequences directly to pre-allocated tensors  
        src_batch_X[i, :len(src_X)] = torch.tensor(src_X, dtype=torch.long, device=device)
        tgt_batch_X[i, :len(tgt_X)] = torch.tensor(tgt_X, dtype=torch.long, device=device)
        tgt_batch_y[i, :len(tgt_y)] = torch.tensor(tgt_y, dtype=torch.long, device=device)

    BatchTensors = namedtuple('BatchTensors',
                              ['src_batch_X', 'tgt_batch_X', 'tgt_batch_y'])
    return BatchTensors(src_batch_X, tgt_batch_X, tgt_batch_y)


def run(config_file: Path):
    config = Config(config_file=config_file)
    transformer, ds = build_transformer(config)
    train_data_loader = DataLoader(
        dataset=ds.dataset,
        batch_size=config.training.batch_size,
        collate_fn=lambda b: collate_fn(
            b,
            pad_id=ds.tokenizer.pad_id,
            bos_id=ds.tokenizer.bos_id,
            eos_id=ds.tokenizer.eos_id,
            max_seq_len=config.model.max_seq_len,
            device=transformer.device
        )
    )
    for batch in train_data_loader:
        print(f"Batch shapes on device {transformer.device}:")
        print(f"  Source: {batch.src_batch_X.shape} - Device: {batch.src_batch_X.device}")
        print(f"  Target Input: {batch.tgt_batch_X.shape} - Device: {batch.tgt_batch_X.device}")
        print(f"  Target Output: {batch.tgt_batch_y.shape} - Device: {batch.tgt_batch_y.device}")
        
        # Test forward pass to verify device compatibility
        print(f"\nTesting forward pass...")
        try:
            with torch.no_grad():  # No gradients needed for testing
                logits = transformer.forward(batch.src_batch_X, batch.tgt_batch_X)
                print(f"Forward pass successful!")
                print(f"  Output logits shape: {logits.shape} - Device: {logits.device}")
                print(f"  Expected shape: [batch_size={batch.src_batch_X.shape[0]}, seq_len={batch.src_batch_X.shape[1]}, vocab_size={config.model.vocab_size}]")
        except Exception as e:
            print(f"Forward pass failed: {e}")
        
        break

