"""
Run module for the transformer project.

This module can be used for running various operations related to the transformer model.
For testing the model, please use the test suite in the tests/ directory.
"""

from model import build_transformer
from config import Config
from pathlib import Path
import torch
import logging
from utils import get_dataloader, BatchTensors
from train import train_batch_CE
from loss import get_loss_function

logger = logging.getLogger(__name__)


# The test_run function has been moved to tests/test_transformer.py
# Use the proper test suite for testing the transformer model

