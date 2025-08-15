from tokenizer.word import WordTokenizer
from pathlib import Path
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from codetiming import Timer
import torch
from torch.utils.data import IterableDataset
from typing import Union
import logging
from typing import Iterator, Tuple
from sentencepiece import SentencePieceProcessor

logger = logging.getLogger(__name__)

@Timer(name="load_data", text="Loading data took {:.2f} seconds")
def get_dataset(data_path: Path, validation: object = True) -> tuple:
    """
    Load the WMT14 German-English dataset from the specified path. Optionally, the
    validation dataset can also be included. The data will be loaded from the
    cache if it exists; otherwise, it will be downloaded.

    :param data_path: Path to the cache directory for the dataset.
    :type data_path: Path
    :param validation: Whether to include the validation dataset. Defaults to
        True.
    :type validation: bool
    :return: If validation is True, returns a tuple containing the training,
        validation, and test datasets. If False, returns a tuple containing the
        training and test datasets.
    :rtype: tuple
    """
    dataset = load_dataset("wmt/wmt14", "de-en",
                 cache_dir=str(data_path),
                 download_mode="reuse_dataset_if_exists")
    if validation:
        return dataset["train"], dataset["validation"], dataset["test"]
    return dataset["train"], None, dataset["test"]


class TransformerDataset(IterableDataset):
    """
    TransformerDataset is an iterable dataset designed for tokenizing and processing
    language translation data efficiently. It takes an existing dataset, tokenizes
    the input and target text using a specified tokenizer, and ensures the sequence
    length is restricted to a defined maximum.

    This dataset is particularly useful for preparing data for training machine
    translation models. It supports lazy iteration over the dataset, converting text
    samples into tensors of tokenized data, which can then be directly used as
    inputs to models.

    :ivar tokenizer: Tokenizer is used to tokenize input and target texts.
    :type tokenizer: Union[SentencePieceProcessor, WordTokenizer]
    :ivar max_seq_len: Maximum allowed sequence length for tokenized data.
    :type max_seq_len: int
    :ivar dataset: Underlying dataset containing text samples with a translation
                   format.
    :type dataset: Dataset
    """
    def __init__(self, dataset: Dataset,
                 tokenizer: Union[SentencePieceProcessor, WordTokenizer],
                 max_seq_len: int):
        self.tokenizer: Union[SentencePieceProcessor, WordTokenizer] = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = dataset


        # Get language keys
        first_sample = next(iter(dataset))
        lang_keys = list(first_sample["translation"].keys())
        self.tokenizer.train(dataset, lang_keys)


    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for sample in self.dataset:
            src_text = sample["translation"]["en"]
            tgt_text = sample["translation"]["de"]

            src_tokens, tgt_tokens = self.tokenizer.encode(src_text, tgt_text, self.max_seq_len)
            if src_tokens and tgt_tokens:
                yield src_tokens, tgt_tokens
