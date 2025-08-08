import click
from pathlib import Path
from run import run
import logging


class LogColors:
    BLUE = '\033[94m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color_map = {
            logging.INFO: LogColors.BLUE,
            logging.WARNING: LogColors.YELLOW,
            logging.ERROR: LogColors.RED,
        }
        color = color_map.get(record.levelno, '')
        record.color_on = color
        record.color_off = LogColors.RESET
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(
    ColoredFormatter('%(color_on)s%(asctime)s - %(levelname)s - %(message)s%(color_off)s')
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)



@click.command()
@click.option('--config',
              type=click.Path(exists=True, path_type=Path),
              default='config.toml',
              help='Path to the configuration file')
def main(config: Path):
    logging.info(f"Running transformer with config {config}")
    run(config_file=config)

if __name__ == '__main__':
    main()