r"""This module implements some helper functions to log information."""

__all__ = ["log_run_info"]

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import gravitorch
from gravitorch.utils.path import get_original_cwd

logger = logging.getLogger(__name__)

# ASCII logo generated from http://patorjk.com/software/taag/#p=testall&f=AMC%203%20Line&t=gravitorch
MLTORCH_LOGO = rf"""

___  ___ _    _____              _
|  \/  || |  |_   _|            | |
| .  . || |    | | ___  _ __ ___| |__
| |\/| || |    | |/ _ \| '__/ __| '_ \
| |  | || |____| | (_) | | | (__| | | |
\_|  |_/\_____/\_/\___/|_|  \___|_| |_|

version: {gravitorch.__version__}"""


def log_run_info(config: DictConfig) -> None:
    """Log some information about the current run.

    Args:
        config (``omegaconf.DictConfig``): Specifies the config of
            the run.
    """
    logger.info(MLTORCH_LOGO)
    logger.info("Original working directory: %s", get_original_cwd())
    logger.info("Current working directory: %s", Path.cwd())
    logger.info("Config:\n%s", OmegaConf.to_yaml(config))
