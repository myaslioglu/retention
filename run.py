import tiktoken

from dataset import load_data
from pathlib import Path
from config import Config
from tokenizer import WordTokenizer
from dataset import TinyStoryDataset
import logging
from transformer.model_encoder import get_encoder



def get_tokenizer(conf: Config, data: str):
    tokenizer_kind = conf.tokenizer.kind.lower()
    if tokenizer_kind == 'tiktoken':
        model = conf.tokenizer.model.lower()
        tk = tiktoken.encoding_for_model(model)
    elif tokenizer_kind == 'custom':
        tk = WordTokenizer(vocab_size=conf.model.vocab_size)
        tk.build_vocab(text=data)
    else:
        raise NotImplementedError(f'🚫Tokenizer of kind: {tokenizer_kind} is not supported! 🚫')
    return tk


def run(config_file: Path):
    config = Config(config_file=config_file)

    # Load the data
    raw_text = load_data(Path(config.dataset.path))

    # Get the Tokenizer
    tokenizer = get_tokenizer(conf=config, data=raw_text)

    # Get the pytorch dataset
    dataset = TinyStoryDataset(config.model.seq_len, tokenizer)
    logging.info(f"Dataset vocab size {dataset.vocab_size}")

    # Tokenize the dataset
    n_tokens: int = dataset.tokenize(raw_data=raw_text)
    logging.info(f"Data converted into {n_tokens} tokens")

    # Get the encoder
    encoder_layers = get_encoder(conf=config, dataset=dataset)
    print(encoder_layers)