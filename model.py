from pathlib import Path
from typing import Union, Tuple
import logging
import gc

import humanize
import psutil
import torch

from config import Config
from utils import get_device
from dataset import TransformerDataset, get_dataset, Dataset
from tokenizer.tokenizer import get_tokenizer

from arch.encoder.model import get_encoder, Encoder
from arch.decoder.model import get_decoder, Decoder
from arch.classifier import get_classifier, Classifier

logger = logging.getLogger(__name__)


def _ensure_attention_kinds(conf: Config) -> None:
    """
    Ensure attention kind flags exist on config with safe defaults and consistency.
    - self_attention_kind: used for encoder self-attn and decoder self-attn (default: 'retention')
    - cross_attention_kind: used for decoder cross-attn over encoder (default: 'mha')
    """
    m = getattr(conf, "model", None)
    if m is None:
        raise ValueError("Config missing .model")

    # Backward-compat: accept a single model.attention_kind and map it to self-attention.
    attn_single = getattr(m, "attention_kind", None)

    self_kind = getattr(m, "self_attention_kind", None) or attn_single or "retention"
    cross_kind = getattr(m, "cross_attention_kind", None) or "mha"

    def _norm(k: str) -> str:
        k = (k or "").lower()
        aliases = {
            "multihead": "mha",
            "multi_head": "mha",
            "msr": "retention",
            "retnet": "retention",
        }
        return aliases.get(k, k)

    self_kind = _norm(self_kind)
    cross_kind = _norm(cross_kind)

    valid = {"mha", "retention"}
    if self_kind not in valid or cross_kind not in valid:
        raise ValueError(
            f"Unknown attention kind(s): self={self_kind}, cross={cross_kind}. "
            f"Valid: {sorted(valid)}"
        )

    m.self_attention_kind = self_kind
    m.cross_attention_kind = cross_kind

    logger.info(
        "Attention kinds -> self: %s | cross: %s",
        m.self_attention_kind,
        m.cross_attention_kind,
    )


def init_dataset(tokenizer, config: Config) -> TransformerDataset:
    """
    Initialize the training dataset with streaming support for memory efficiency.
    Loads the WMT14 German-English dataset in streaming mode to avoid RAM overflow.
    """
    dataset_path = Path(config.dataset.path)
    train, _, _ = get_dataset(dataset_path, validation=False, streaming=True)
    train_ds = TransformerDataset(train, tokenizer, config.model.max_seq_len)
    return train_ds


def init_tokenizer(config: Config):
    """
    Initialize a tokenizer with configuration-specific model paths (no collisions
    across experiments). Creates a directory:
        <base>/<sample_size>-<algorithm>-<vocab_size>/tokenizer
    """
    kind = config.tokenizer.kind
    vocab_size = config.model.vocab_size
    base_path = Path(config.tokenizer.model)

    config_dir = f"{config.tokenizer.sample_size}-{config.tokenizer.algorithm}-{vocab_size}"
    tk_model_path = base_path / config_dir / "tokenizer"

    logger.info("Using tokenizer path: %s", tk_model_path)

    if config.tokenizer.recreate:
        from shutil import rmtree
        rmtree(tk_model_path.parent, ignore_errors=True)
        logger.info("Removed existing tokenizer at %s", tk_model_path.parent)

    return get_tokenizer(
        tokenizer_kind=kind,
        model_path=tk_model_path,
        vocab_size=vocab_size,
        algorithm=config.tokenizer.algorithm,
        sample_size=config.tokenizer.sample_size,
    )


class TransformerModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, classifier: Classifier, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.device = device

    def forward(
        self,
        src_batch: torch.Tensor,
        tgt_batch: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through encoder -> decoder (with cross-attn) -> classifier.
        Shapes:
            src_batch: [B, S]
            tgt_batch: [B, T]
        Returns:
            logits: [B, T, vocab_size]
        """
        encoder_output = self.encoder(src_batch, src_pad_mask)
        decoder_output = self.decoder(tgt_batch, tgt_pad_mask, encoder_output)
        logits = self.classifier(decoder_output)
        return logits

    def to(self, device: torch.device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.classifier = self.classifier.to(device)
        self.device = device
        return self

    def get_layer(self, layer_chain: str) -> Union[torch.nn.Module, None]:
        parent = self
        model = None
        for layer in layer_chain.split("."):
            model = getattr(parent, layer, None)
            if model is None:
                break
            parent = model
        if model is None:
            logger.error("Layer chain %s not found in model", layer_chain)
            return None
        return model

    @staticmethod
    def _eval_initializer(init_method: str):
        match init_method:
            case "xavier_uniform":
                return torch.nn.init.xavier_uniform_
            case "xavier_normal":
                return torch.nn.init.xavier_normal_
            case "kaiming_uniform":
                return torch.nn.init.kaiming_uniform_
            case "kaiming_normal":
                return torch.nn.init.kaiming_normal_
        logger.warning("Unknown init method %s, defaulting to xavier_uniform", init_method)
        return torch.nn.init.xavier_uniform_

    def initialize_weights_(self, init_method: str, init_bias: bool):
        initializer = TransformerModel._eval_initializer(init_method)
        self.encoder._init_layers(initializer, init_bias)
        self.decoder._init_layers(initializer, init_bias)
        logger.debug("Initialized encoder/decoder layers")


def build_transformer(config: Config) -> Tuple[TransformerModel, Dataset]:
    """
    Build tokenizer, dataset, encoder/decoder/classifier; move to device.
    Adds explicit attention-kind wiring for retention/MHA.
    """
    # Log initial memory
    mem_before = psutil.virtual_memory()
    logger.info(
        "Memory before model creation: %.1f%% used (%s/%s)",
        mem_before.percent,
        humanize.naturalsize(mem_before.used),
        humanize.naturalsize(mem_before.total),
    )

    device = get_device(config)

    # Validate and normalize attention kinds (RETENTION by default for self, MHA for cross)
    _ensure_attention_kinds(config)

    # Tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = init_tokenizer(config=config)
    mem_after_tok = psutil.virtual_memory()
    logger.info(
        "Memory after tokenizer: %.1f%% used (%s/%s)",
        mem_after_tok.percent,
        humanize.naturalsize(mem_after_tok.used),
        humanize.naturalsize(mem_after_tok.total),
    )

    # Dataset
    logger.info("Initializing dataset...")
    dataset = init_dataset(config=config, tokenizer=tokenizer)
    # Align vocab to tokenizer
    config.model.vocab_size = dataset.tokenizer.n_vocab
    mem_after_ds = psutil.virtual_memory()
    logger.info(
        "Memory after dataset: %.1f%% used (%s/%s)",
        mem_after_ds.percent,
        humanize.naturalsize(mem_after_ds.used),
        humanize.naturalsize(mem_after_ds.total),
    )

    # Encoder
    logger.info(
        "Initializing encoder (self-attention: %s)...",
        config.model.self_attention_kind,
    )
    encoder_model = get_encoder(conf=config).to(device)
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Decoder
    logger.info(
        "Initializing decoder (self-attention: %s, cross-attention: %s)...",
        config.model.self_attention_kind,
        config.model.cross_attention_kind,
    )
    decoder_model = get_decoder(conf=config).to(device)
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Classifier
    logger.info("Initializing classifier head...")
    classifier_model = get_classifier(conf=config).to(device)

    # Params + memory
    total_params = (
        sum(p.numel() for p in encoder_model.parameters())
        + sum(p.numel() for p in decoder_model.parameters())
        + sum(p.numel() for p in classifier_model.parameters())
    )
    logger.info("Total model parameters: %s", humanize.intword(total_params))
    logger.info(
        "Approximate model memory usage: %s",
        humanize.naturalsize(total_params * 4),
    )

    mem_final = psutil.virtual_memory()
    logger.info(
        "Memory after model creation: %.1f%% used (%s/%s)",
        mem_final.percent,
        humanize.naturalsize(mem_final.used),
        humanize.naturalsize(mem_final.total),
    )

    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return (
        TransformerModel(
            encoder=encoder_model,
            decoder=decoder_model,
            classifier=classifier_model,
            device=device,
        ),
        Dataset(dataset=dataset, tokenizer=tokenizer),
    )
