import click
from pathlib import Path
import logging
from log_config import setup_logging
from run import run_train


@click.command()
@click.option('--config',
              type=click.Path(exists=True, path_type=Path),
              default='config.toml',
              help='Path to the configuration file')
def main(config: Path):
    """
    Main entry point for transformer training with configurable parameters.
    
    This function sets up logging, validates the configuration file, and initiates
    the transformer training process with the specified configuration.
    
    Args:
        config (Path): Path to the TOML configuration file containing model,
            training, tokenizer, and experiment parameters.
    
    Note:
        This function is decorated with Click to provide a command-line interface.
        It accepts a --config option that defaults to 'config.toml'.
    """
    setup_logging(for_notebook=False)
    logging.info(f"Running transformer with config {config}")
    run_train(config_file=config)

if __name__ == '__main__':
    main()
