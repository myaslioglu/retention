import click
from pathlib import Path
from model import create_model
import logging
from log_config import setup_logging

@click.command()
@click.option('--config',
              type=click.Path(exists=True, path_type=Path),
              default='config.toml',
              help='Path to the configuration file')
def main(config: Path):
    setup_logging(for_notebook=False)
    logging.info(f"Running transformer with config {config}")
    model = create_model(config_file=config)
    logging.info(f"The model is {model}")

if __name__ == '__main__':
    main()