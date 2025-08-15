from model import build_transformer
from config import Config
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from collections import namedtuple


def collate_fn(batch, pad_id: int,  bos_id: int, eos_id: int, max_seq_len: int):
    src_batch_X = []
    tgt_batch_X = []
    tgt_batch_y = []

    for src_tkn, tgt_tkn in batch:
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

        # Convert to tensors
        src_batch_X.append(torch.tensor(src_X, dtype=torch.long))
        tgt_batch_X.append(torch.tensor(tgt_X, dtype=torch.long))
        tgt_batch_y.append(torch.tensor(tgt_y, dtype=torch.long))

    # Stack into batch tensors
    src_batch_X_tensor = torch.stack(src_batch_X)  # [batch_size, max_seq_len]
    tgt_batch_X_tensor = torch.stack(tgt_batch_X)  # [batch_size, max_seq_len]
    tgt_batch_y_tensor = torch.stack(tgt_batch_y)  # [batch_size, max_seq_len]

    BatchTensors = namedtuple('BatchTensors',
                              ['src_batch_X', 'tgt_batch_X', 'tgt_batch_y'])
    return BatchTensors(src_batch_X_tensor, tgt_batch_X_tensor, tgt_batch_y_tensor)


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
            max_seq_len=config.model.max_seq_len
        )
    )
    for batch in train_data_loader:
        print(batch.src_batch_X.shape)
        print(batch.tgt_batch_X.shape)
        print(batch.tgt_batch_y.shape)
        break

