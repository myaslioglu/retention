from pathlib import Path
from typing import Union
import torch
from config import Config
from dataset import TransformerDataset, get_dataset
import logging
from tokenizer.tokenizer import get_tokenizer
from arch.encoder.model import get_encoder, Encoder
from arch.decoder.model import get_decoder, Decoder
from arch.classifier import get_classifier, Classifier
from dataclasses import dataclass
from shutil import rmtree
import humanize
from utils import get_device
from dataset import Dataset

logger = logging.getLogger(__name__)

@dataclass
class TransformerModel:
    encoder: Encoder
    decoder: Decoder
    classifier: Classifier
    device: torch.device
    
    def forward(self, src_batch: torch.Tensor, tgt_batch: torch.Tensor, src_pad_mask: torch.Tensor, tgt_pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete transformer model.
        
        :param src_batch: Source sequences [batch_size, seq_len]
        :param tgt_batch: Target sequences [batch_size, seq_len]
        :return: Output logits [batch_size, seq_len, vocab_size]
        """
        # Ensure inputs are on the correct device
        src_batch = src_batch.to(self.device)
        tgt_batch = tgt_batch.to(self.device)
        
        # Encoder forward pass
        encoder_output = self.encoder(src_batch, src_pad_mask)

        # Decoder forward pass with cross-attention to encoder output
        decoder_output = self.decoder(tgt_batch, tgt_pad_mask, encoder_output)
        
        # Classifier forward pass to get vocabulary logits
        logits = self.classifier(decoder_output)
        
        return logits
    
    def to(self, device: torch.device):
        """Move all model components to the specified device."""
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.classifier = self.classifier.to(device)
        self.device = device
        return self


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




def build_transformer(config: Config) -> tuple[TransformerModel, Dataset]:
    """
    Builds and initializes the main transformer model and its associated dataset.
    This function performs the following steps:
    - Determines the best available device (CUDA/CPU)
    - Initializes the tokenizer based on the provided configuration.
    - Prepares the dataset, ensuring the tokenizer and dataset align in terms of
      vocabulary size.
    - Initializes encoder, decoder, and classifier modules to construct the model.
    - Moves all model components to the selected device.

    :param config: Configuration object containing parameters for initializing
        the tokenizer, dataset, and model components.
    :type config: Config
    :return: A tuple containing the initialized `Model` and `Dataset` objects.
    :rtype: tuple[Model, Dataset]
    """
    # Determine the device to use
    device = get_device(config)
    # Get the tokenizer
    tokenizer = init_tokenizer(config=config)

    # Get the dataset
    dataset = init_dataset(config=config, tokenizer=tokenizer)
    # This is needed because the tokenizer can have vocab size != desired vocab size
    config.model.vocab_size = dataset.tokenizer.n_vocab

    # Get the encoder
    encoder_model = get_encoder(conf=config).to(device)
    logger.debug("Encoder initialized and moved to %s: %s", device, encoder_model)

    # Get the decoder
    decoder_model = get_decoder(conf=config).to(device)
    logger.debug("Decoder initialized and moved to %s: %s", device, decoder_model)

    # Get the Classifier Head
    classifier_model = get_classifier(conf=config).to(device)
    logger.debug("Classifier initialized and moved to %s", device)

    # Log total parameters
    total_params = sum(p.numel() for p in encoder_model.parameters()) + \
                   sum(p.numel() for p in decoder_model.parameters()) + \
                   sum(p.numel() for p in classifier_model.parameters())
    logger.info(f"Total model parameters: {humanize.intword(total_params)}")

    return TransformerModel(
        encoder=encoder_model,
        decoder=decoder_model,
        classifier=classifier_model,
        device=device
    ), Dataset(
        dataset=dataset,
        tokenizer=tokenizer
    )
