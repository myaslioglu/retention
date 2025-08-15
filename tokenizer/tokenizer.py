from .sentencepiece import SentencePieceTokenizer
from .word import WordTokenizer
from pathlib import Path

def get_tokenizer(tokenizer_kind: str, model_path: Path,
                  vocab_size: int, algorithm: str = 'bpe',
                  sample_size: int=200_000,
                  **kwargs):
    """Return a tokenizer based on config; builds vocab for the custom tokenizer."""
    if tokenizer_kind == 'sentencepiece':
        tk = SentencePieceTokenizer(
            model_path=model_path,
            vocab_size=vocab_size,
            algorithm=algorithm,
            sample_size=sample_size,
            add_special_tokens=True,
            **kwargs
        )
    elif tokenizer_kind == 'custom':
        tk = WordTokenizer(vocab_size=vocab_size)
    else:
        raise NotImplementedError(
            f'🚫Tokenizer of kind: {tokenizer_kind} is not supported! 🚫'
        )
    return tk
