"""
This module can be used for running various operations related to the transformer model.
For testing the model, please use the test suite in the tests/ directory.
"""

from model import build_transformer
from config import Config
from pathlib import Path
import logging
from utils import get_dataloader
from train import train_epoch_avg_CE
from loss import get_loss_function
from wnb import ExperimentTracker

logger = logging.getLogger(__name__)


def run_train(config_file: Path):
    config = Config(config_file=config_file)

    transformer, dataset = build_transformer(config)

    train_data_loader = get_dataloader(dataset, config)

    loss_fn = get_loss_function(config, dataset.tokenizer.pad_id)

    # Wandb requires metrics to be in dict format
    wandb_config = {
        'batch_size': config.training.batch_size,
        'd_model': config.model.hidden_size,
        'num_heads': config.model.n_heads,
        'num_layers': config.model.n_layers,
        'max_seq_len': config.model.max_seq_len,
    }

    # Experiment tracking can be off if running offline without internet
    if config.experiment.active:
        logger.info("Experiment tracking is enabled")
        with ExperimentTracker(config.experiment.name, wandb_config, config_file=config_file) as run:
            avg_batch_loss = train_epoch_avg_CE(model=transformer,
                                                train_data_loader=train_data_loader,
                                                loss_fn=loss_fn,
                                                wandb_run=run)
            run.log({"epoch_avg_loss": avg_batch_loss.item()})
            logger.info(f"Average training loss in one epoch: {avg_batch_loss.item(): .5f}")
    else:
        logger.info("Experiment tracking is disabled")
        avg_batch_loss = train_epoch_avg_CE(model=transformer,
                                            train_data_loader=train_data_loader,
                                            loss_fn=loss_fn,
                                            wandb_run=None)
        logger.info(f"Average training loss in one epoch: {avg_batch_loss.item(): .5f}")
