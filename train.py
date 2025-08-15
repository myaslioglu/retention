import logging
import torch
from model import Model

logger = logging.getLogger(__name__)

def train_with_BLEU(model: Model, dataset, batch_size: int, epoch: int):
    """
    """
    count = 0
    buffer = []
    for data in dataset:
        if count == epoch:
            break
        buffer.append(data)
        if len(buffer) == batch_size:
            count += 1
            logger.info(f"Processing batch {count}")
            train_one(model, torch.stack(buffer))
            buffer = []

def train_one(model: Model, src_tkn: torch.Tensor, tgt_tkn: torch.Tensor) -> torch.Tensor:
    """
    Trains one step in an encoder-decoder model by passing the source and target tokens
    through the encoder, decoder, and classifier components of the model. The method assumes
    that the model is composed of an encoder, a decoder, and a classifier, which work in tandem
    to process the input and produce an output.

    :param model: The encoder-decoder model consists of an encoder, decoder,
        and classifier.
    :type model: Model
    :param src_tkn: The input source token tensor to be processed by the encoder.
    :type src_tkn: torch.Tensor
    :param tgt_tkn: The target token tensor to be processed by the decoder, with
        dependency on the output of the encoder.
    :type tgt_tkn: torch.Tensor
    :return: The classifier's output after processing the decoder's output.
    :rtype: Any
    """
    encoder_model = model.encoder
    decoder_model = model.decoder
    classifier_model = model.classifier

    encoder_output = encoder_model(src_tkn)
    decoder_output = decoder_model(tgt_tkn, encoder_output)
    output = classifier_model(decoder_output)
    return output
