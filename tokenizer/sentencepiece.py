import logging
import sentencepiece as spm
from pathlib import Path
import tempfile

from datasets.arrow_dataset import Dataset

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    def __init__(self, model_path: Path,
                 vocab_size: int,
                 algorithm: str,
                 sample_size: int,
                 add_special_tokens: bool,
                 **kwargs):
        self.algorithm = algorithm
        self.sample_size = sample_size
        self.model_path = model_path
        self.vocab_size = vocab_size

        self.BOS = kwargs.get("BOS", "<s>")
        self.EOS = kwargs.get("EOS", "</s>")
        self.PAD = kwargs.get("PAD", "<pad>")
        self.UNK = kwargs.get("UNK", "<unk>")
        self._model = None
        self.add_special_tokens = add_special_tokens
        self.actual_vocab_size = None  # This can differ from the provided vocab_size

    @property
    def model(self):
        model = spm.SentencePieceProcessor()
        logger.info("Loading SentencePiece model from %s", self.model_path)
        model.Load(str(self.model_path.with_suffix(".model")))
        return model, model.GetPieceSize()

    def train(self, dataset: Dataset, lang_keys: list):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.model_path.with_suffix(".model").exists():
            all_samples = []
            for i, sample in enumerate(dataset):
                if i >= self.sample_size:
                    break
                for lang_key in lang_keys:
                    all_samples.append(sample["translation"][lang_key])

            # Write combined training data and tran the model
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                             delete_on_close=True) as f:
                f.write('\n'.join(all_samples))
                spm.SentencePieceTrainer.Train(
                    input=f.name,
                    model_prefix=str(self.model_path),
                    vocab_size=self.vocab_size,
                    character_coverage=0.9995,  # High coverage for both languages
                    model_type=self.algorithm,
                    pad_id=0,
                    unk_id=1,
                    bos_id=2,
                    eos_id=3,
                    pad_piece=self.PAD,
                    unk_piece=self.UNK,
                    bos_piece=self.BOS,
                    eos_piece=self.EOS,
                    user_defined_symbols=[],  # // TODO: Add user tokens later
                    shuffle_input_sentence=True,  # Better statistical coverage
                )
        self._model, self.actual_vocab_size = self.model
        if self.actual_vocab_size != self.vocab_size:
            logger.warning("Actual vocab size (%d) does not match the provided one (%d)",
                           self.actual_vocab_size, self.vocab_size)
        logger.info("SentencePiece model loaded successfully!")

    @property
    def n_vocab(self):
        return self.actual_vocab_size

    def encode(self, src_txt: str, tgt_txt: str, stride: int) -> tuple[list[int], list[int]] | tuple[None, None]:
        if not self._model:
            logger.error("Please train the sentencepiece tokenizer first")
            return None, None
        src_tokens = self._model.encode(src_txt, out_type=int)
        tgt_tokens = self._model.encode(tgt_txt, out_type=int)

        bos_id = self._model.bos_id()
        eos_id = self._model.eos_id()
        pad_id = self._model.pad_id()

        if self.add_special_tokens:
            src_tokens = [bos_id] + src_tokens[:stride - 2] + [eos_id]
            tgt_tokens = [bos_id] + tgt_tokens[:stride - 2] + [eos_id]

        # Add the padding to the tokens
        src_tokens += [pad_id] * (stride - len(src_tokens))
        tgt_tokens += [pad_id] * (stride - len(tgt_tokens))

        return src_tokens, tgt_tokens

