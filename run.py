from model import build_transformer
from config import Config
from pathlib import Path
from itertools import islice as take
from train import train_one

def run(config_file: Path):
    config = Config(config_file=config_file)
    transformer, ds = build_transformer(config)
    for src_tkn, tgt_tkn in take(ds.dataset, 5):
        output = train_one(transformer, src_tkn, tgt_tkn)
        print(f"Output Shape: {output.shape}")


