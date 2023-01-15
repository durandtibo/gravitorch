r"""This module implements some helper functions to log information."""

__all__ = ["log_run_info"]

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import gravitorch
from gravitorch.utils.path import get_original_cwd

logger = logging.getLogger(__name__)

# ASCII logo generated from https://ascii.co.uk/art/cats
GRAVITORCH_LOGO = rf"""

        /\_/\  /\          /\_/\  /\          /\_/\  /\          /\_/\  /\
       / o o \ \ \        / o o \ \ \        / o o \ \ \        / o o \ \ \
      /   Y   \/ /       /   Y   \/ /       /   Y   \/ /       /   Y   \/ /
     /         \/       /         \/       /         \/       /         \/
     \ | | | | /        \ | | | | /        \ | | | | /        \ | | | | /
      `|_|-|_|'          `|_|-|_|'          `|_|-|_|'          `|_|-|_|'

version: {gravitorch.__version__}"""


def log_run_info(config: DictConfig) -> None:
    """Log some information about the current run.

    Args:
        config (``omegaconf.DictConfig``): Specifies the config of
            the run.
    """
    logger.info(GRAVITORCH_LOGO)
    logger.info("Original working directory: %s", get_original_cwd())
    logger.info("Current working directory: %s", Path.cwd())
    logger.info("Config:\n%s", OmegaConf.to_yaml(config))
