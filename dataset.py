from torch.utils.data import Dataset
from pathlib import Path
from datasets import load_dataset
from codetiming import Timer

def load_data(data_path: Path) -> str:
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


class TinyStoryDataset(Dataset):
    """
    This uses the data dir to load the dataset
    of TinyStory.txt into the dataloader.
    """
    def __init__(self, seq_len: int, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.token_ids = None

    @Timer(name="tokenize", text="Tokenization took {:.2f} seconds")
    def tokenize(self, raw_data: str):
        """
        Tokenize the raw_data and return total
        length of tokens
        :param raw_data: Raw text data to tokenize
        :return: Total number of tokens produced
        """
        # TODO: We can create concurrent workers for processing large input
        token_ids = self.tokenizer.encode(raw_data)
        self.token_ids = token_ids
        return len(self.token_ids)

    @property
    def vocab_size(self):
        """
        Get the vocabulary size of the tokenizer
        :return: Vocab size
        """
        return self.tokenizer.n_vocab