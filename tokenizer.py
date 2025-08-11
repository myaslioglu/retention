import re
import logging
from tqdm import tqdm
import tiktoken
from typing import Union

logger = logging.getLogger(__name__)

class WordTokenizer:
    def __init__(self, vocab_size: int, unknown_token: str = None):
        if not unknown_token:
            unknown_token = '<UNK>'
        self.unknown_token = unknown_token
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_to_id[self.unknown_token] = 0
        self.id_to_word[0] = self.unknown_token
        self.next_id = 1 # Reserve 0 for UNKNOWN words
        self._n_vocab = vocab_size

    def build_vocab(self, text):
        words = WordTokenizer.extract_words(text)
        for word in tqdm(words, desc="Building vocabulary:"):
            if word not in self.word_to_id and len(self.word_to_id) < self._n_vocab:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1

    def encode(self, text):
        self.build_vocab(text)
        if len(self.word_to_id) < 2 or len(self.id_to_word) < 2:
            logger.error("Please build the vocab first")
            return None
        words = WordTokenizer.extract_words(text)
        return [self.word_to_id.get(word, self.word_to_id[self.unknown_token]) for word in words]

    def decode(self, ids):
        if len(self.word_to_id) < 2 or len(self.id_to_word) < 2:
            logger.error("Please build the vocab first")
            return None
        return " ".join(self.id_to_word[i] for i in ids)

    @staticmethod
    def extract_words(text):
        return re.findall(r"\b\w+\b", text.lower())

    @property
    def n_vocab(self):
        return self._n_vocab


def get_tokenizer(tokenizer_kind: str, tokenizer_model: str, vocab_size: Union[int, None] = None):
    """Return a tokenizer based on config; builds vocab for the custom tokenizer."""
    if tokenizer_kind == 'tiktoken':
        tk = tiktoken.encoding_for_model(tokenizer_model)
    elif tokenizer_kind == 'custom':
        if not vocab_size:
            raise AttributeError("Explicit vocab_size is required for custom tokenizer!")
        tk = WordTokenizer(vocab_size=vocab_size)
    else:
        raise NotImplementedError(
            f'🚫Tokenizer of kind: {tokenizer_kind} is not supported! 🚫'
        )
    return tk
