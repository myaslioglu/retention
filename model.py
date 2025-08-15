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
from tokenizer.sentencepiece import SentencePieceTokenizer
from tokenizer.word import WordTokenizer

logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    dataset: TransformerDataset
    tokenizer: Union[SentencePieceTokenizer, WordTokenizer]

@dataclass
class Model:
    encoder: Encoder
    decoder: Decoder
    classifier: Classifier
    device: torch.device
    
    def forward(self, src_batch: torch.Tensor, tgt_batch: torch.Tensor) -> torch.Tensor:
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
        encoder_output = self.encoder(src_batch)
        
        # Decoder forward pass with cross-attention to encoder output
        decoder_output = self.decoder(tgt_batch, encoder_output)
        
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


def build_transformer(config: Config) -> tuple[Model, Dataset]:
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
    logger.info(f"Total model parameters: {total_params:,}")

    return Model(
        encoder=encoder_model,
        decoder=decoder_model,
        classifier=classifier_model,
        device=device
    ), Dataset(
        dataset=dataset,
        tokenizer=tokenizer
    )
