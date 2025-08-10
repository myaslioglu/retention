from dataset import load_data
from pathlib import Path
from config import Config
from dataset import TinyStoryDataset
import logging
from training import train
from tokenizer import get_tokenizer
from transformer.encoder.model import get_encoder
from transformer.decoder.model import get_decoder

logger = logging.getLogger(__name__)


def create_model(config_file: Path):
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
    logger.info("Data converted into %d tokens", n_tokens)


    # Get the encoder
    encoder_model = get_encoder(conf=config)
    logger.debug("Encoder initialized: %s", encoder_model)

    # Get the decoder
    decoder_model = get_decoder(conf=config)
    logger.debug("Decoder initialized: %s", decoder_model)

    # Iterate over the dataset
    batch_size: int = config.training.batch_size
    epochs: int = config.training.epochs
    logger.info("Starting training: batch_size=%d, epochs=%d", batch_size, epochs)

    train(encoder_model, decoder_model, dataset, batch_size, epochs)
