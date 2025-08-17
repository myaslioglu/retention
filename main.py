import click
from pathlib import Path
from run import test_run
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
    test_run(config_file=config)

if __name__ == '__main__':
    main()
