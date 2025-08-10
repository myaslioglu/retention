from config import Config
from pathlib import Path
from datasets import load_dataset
from codetiming import Timer
import torch
from torch.utils.data import IterableDataset

@Timer(name="load_data", text="Loading data took {:.2f} seconds")
def load_data(data_path: Path) -> str:
    """
    Loads text data from the specified file path or retrieves the dataset from a remote source and saves
    it locally if the file path does not exist.

    If the file exists, the function reads the text data from the file. If the file does not exist or
    the path is invalid, it loads a specified dataset, extracts text fields, and writes the combined
    text data to the specified file path. The resulting text data is returned in both cases.

    :param data_path: Path object pointing to the file location where the data is to be loaded from or
                     saved to.
    :return: The raw text data as a single string is retrieved from the file or the dataset.
    """
    if data_path.exists() and data_path.is_file():
        with open(data_path, 'r') as f:
            raw_text = f.read()
    else:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        texts = [sample["text"] for sample in dataset]
        raw_text = "\n\n".join(texts)
        with open(data_path, 'w') as f:
            f.write(raw_text)
    return raw_text

class TinyStoryDataset(IterableDataset):
    """
    Creates a dataset for handling tokenized sequences with optional file streaming
    support and in-memory processing.

    The TinyStoryDataset class is designed for iterating over tokenized text sequences.
    It supports both eager tokenization (with data stored in-memory) and lazy streaming
    (using a file). The class implements map-style and iterable-style dataset interfaces
    compatible with PyTorch, providing flexibility for sequence-based data processing.

    :ivar seq_len: The length of each sequence window to be processed or returned.
    :type seq_len: int
    :ivar tokenizer: The tokenizer used to encode raw textual input into token IDs.
    :ivar data_path: The optional path to a file containing raw input data, which can
        be streamed during iteration.
    :type data_path: Path | None
    :ivar stride: The step size for sliding window processing, defaults to `seq_len`
        (non-overlapping windows by default).
    :type stride: int | None
    """
    def __init__(self, config: Config | None = None,
                 seq_len: int | None = None,
                 stride: int | None = None):
        self.config = config
        if config is not None:
            self.seq_len = config.model.seq_len
        elif seq_len is not None:
            self.seq_len = seq_len
        else:
            raise ValueError("Provide either config or seq_len to TinyStoryDataset")
        self.token_ids = None
        self.stride = stride if stride is not None else self.seq_len  # non-overlapping by default
        self._vocab_size: int | None = None

    @Timer(name="tokenize", text="Tokenization took {:.2f} seconds")
    def tokenize(self, tokenizer, data: str, inplace: bool = False):
        """
        Tokenizes the provided data using the specified tokenizer. This method encodes
        the input data into tokens. If the `inplace` parameter is set to True, the
        generated tokens will be stored in the instance's `token_ids` attribute and
        the method will return the count of tokens. Otherwise, it will return the
        list of tokens.

        :param tokenizer: A tokenizer object used for encoding the data.
        :param data: The input string data that needs to be tokenized.
        :param inplace: A boolean flag indicating whether the tokens are set
            directly on the instance or returned. Defaults to False.
        :return: A list of encoded tokens if `inplace` is False, or the
            count of tokens if `inplace` is True.
        """
        tokens = tokenizer.encode(data)
        self._vocab_size = getattr(tokenizer, 'n_vocab', None)
        if self.config is not None and self._vocab_size is not None:
            # keep config in sync when using Config-based setup
            self.config.model.vocab_size = self._vocab_size
        if inplace:
            self.token_ids = tokens
            return len(tokens)
        else:
            return tokens

    def __len__(self) -> int:
        """Length only meaningful for eager mode when token_ids is available."""
        if not self.token_ids:
            return 0
        total = len(self.token_ids) - self.seq_len + 1
        return max(total, 0)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Retrieve a sequence window from the stored token IDs based on the provided index.

        This method allows access to tokenized data in a map-style format, returning
        a slice of token IDs of a fixed sequence length (`seq_len`). If a slice is shorter
        than the specified sequence length, it will be padded with zeros.

        :param index: The starting index for the sequence slice.
        :type index: int
        :return: A tensor representing the tokenized sequence of fixed length
                 (`seq_len`). If the available data is insufficient, the tensor
                 is zero-padded to match the required length.
        :rtype: torch.Tensor
        :raises RuntimeError: If `tokenize(raw_data)` has not been called prior to
                              accessing data via indexing.
        """
        if self.token_ids is None:
            raise RuntimeError("Call tokenize(raw_data) before map-style access or use streaming with data_path")
        start = index
        end = index + self.seq_len
        window = self.token_ids[start:end]
        if len(window) < self.seq_len:
            pad_len = self.seq_len - len(window)
            window = window + [0] * pad_len
        return torch.tensor(window, dtype=torch.long)

    def __iter__(self):
        if self.token_ids is None:
            raise RuntimeError("token_ids are not set; call tokenize(raw_data) before iterating")

        if not isinstance(self.token_ids, list):
            raise TypeError("token_ids must be a list")

        start = 0
        while start + self.seq_len <= len(self.token_ids):
            window = self.token_ids[start : start+self.seq_len]
            yield torch.tensor(window, dtype=torch.long)
            start += self.stride

    @property
    def tokens_ids(self):
        return self.token_ids

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("vocab_size is unknown; call tokenize(...) first")
        return self._vocab_size