import tomllib
from pathlib import Path
from box import Box
import logging

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file: Path):
        self._config = None
        if config_file.is_file():
            with open(config_file, 'rb') as f:
                self._config = Box(tomllib.load(f))
    @property
    def model(self):
        return self._config.model

    @property
    def tokenizer(self):
        return self._config.tokenizer

    @property
    def training(self):
        return self._config.training

    @property
    def dataset(self):
        return self._config.dataset


if __name__ == '__main__':
    config = Config(config_file=Path('config.toml'))
    logger.info("Model.ffwfs = %s", config.model.get('ffwfs'))
