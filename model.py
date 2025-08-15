from pathlib import Path
from config import Config
from dataset import TransformerDataset, get_dataset
import logging
from tokenizer.tokenizer import get_tokenizer
from arch.encoder.model import get_encoder, Encoder
from arch.decoder.model import get_decoder, Decoder
from arch.classifier import get_classifier, Classifier
from dataclasses import dataclass
from shutil import rmtree

logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    dataset: TransformerDataset
    tokenizer: object

@dataclass
class Model:
    encoder: Encoder
    decoder: Decoder
    classifier: Classifier


def init_dataset(tokenizer, config: Config):
    dataset_path = Path(config.dataset.path)

    # // TODO: Cater for test and validation datasets
    train, _, _ = get_dataset(dataset_path, validation=False)
    train_ds = TransformerDataset(train, tokenizer, config.model.max_seq_len)
    return train_ds


def init_tokenizer(config: Config):
    kind = config.tokenizer.kind
    vocab_size = config.model.vocab_size
    tk_model_path = Path(config.tokenizer.model)
    if config.tokenizer.recreate:
        # Delete the existing model and vocab files
        rmtree(tk_model_path.parent, ignore_errors=True)
    return get_tokenizer(kind, tk_model_path, vocab_size)


def build_transformer(config: Config) -> tuple[Model, Dataset]:
    """
    Builds and initializes the main transformer model and its associated dataset.
    This function performs the following steps:
    - Initializes the tokenizer based on the provided configuration.
    - Prepares the dataset, ensuring the tokenizer and dataset align in terms of
      vocabulary size.
    - Initializes encoder, decoder, and classifier modules to construct the model.

    :param config: Configuration object containing parameters for initializing
        the tokenizer, dataset, and model components.
    :type config: Config
    :return: A tuple containing the initialized `Model` and `Dataset` objects.
    :rtype: tuple[Model, Dataset]
    """
    # Get the tokenizer
    tokenizer = init_tokenizer(config=config)

    # Get the dataset
    dataset = init_dataset(config=config, tokenizer=tokenizer)
    # This is needed because the tokenizer can have vocab size != desired vocab size
    config.model.vocab_size = dataset.tokenizer.n_vocab

    # Get the encoder
    encoder_model = get_encoder(conf=config)
    logger.debug("Encoder initialized: %s", encoder_model)

    # Get the decoder
    decoder_model = get_decoder(conf=config)
    logger.debug("Decoder initialized: %s", decoder_model)

    # Get the Classifier Head
    classifier_model = get_classifier(conf=config)

    return Model(
        encoder=encoder_model,
        decoder=decoder_model,
        classifier=classifier_model
    ), Dataset(
        dataset=dataset,
        tokenizer=tokenizer
    )
