from dataset import load_data
from pathlib import Path
from config import Config
from dataset import TinyStoryDataset
import logging
from training import train
from transformer.model_encoder import get_encoder
from tokenizer import get_tokenizer


def run(config_file: Path):
    config = Config(config_file=config_file)

    # Get the Tokenizer
    tokenizer_kind: str = config.tokenizer.kind
    tokenizer_model: str = config.tokenizer.model
    vocab_size: int = config.model.vocab_size # Needed ONLY for custom tokenizer
    tokenizer = get_tokenizer(tokenizer_kind, tokenizer_model, vocab_size)

    # Get the pytorch dataset
    dataset = TinyStoryDataset(config)

    # Tokenize the data
    DATAPATH = Path(config.dataset.path)
    data = load_data(DATAPATH)
    n_tokens: int = dataset.tokenize(tokenizer, data, inplace=True)
    logging.info(f"Data converted into {n_tokens} tokens")


    # Get the encoder
    encoder = get_encoder(conf=config)
    print(encoder)

    # Iterate over the dataset
    batch_size: int = config.training.batch_size
    epochs: int = config.training.epochs
    train(encoder, dataset, batch_size, epochs)
