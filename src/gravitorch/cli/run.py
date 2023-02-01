r"""This module contains the entrypoint to run experiments.

Example usage:

.. code-block::

python -m gravitorch.cli.run -cd=examples/classification
-cn=mlp_classification_sgd engine.state.max_epochs=2  # noqa: E501,B950
"""

__all__ = ["main", "run_cli"]

import logging
from typing import Any

import hydra
from hya import register_resolvers
from omegaconf import DictConfig, OmegaConf

from gravitorch import constants as ct
from gravitorch.runners import setup_runner
from gravitorch.utils.info import log_run_info
from gravitorch.utils.timing import timeblock

logger = logging.getLogger(__name__)


def main(config: dict[str, Any]) -> None:
    r"""Initializes a runner given its configuration and executes its logic.

    Args:
        config (dict): Specifies the dictionary with the configuration
            of the runner. This dictionary has to have a key
            ``'runner'``.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.cli.run import main
        >>> config = {
        ...     '_target_': 'gravitorch.runners.TrainingRunner',
        ...     'engine': {'_target_': 'gravitorch.engines.AlphaEngine', ...},
        ...     'exp_tracker': {'_target_': 'gravitorch.utils.exp_trackers.NoOpExpTracker', ...},
        ... }
        >>> main(config)
    """
    with timeblock("Total time of the run: {time}"):
        logger.info("Creating runner...")
        runner = setup_runner(config[ct.RUNNER])
        logger.info(f"runner:\n{runner}")
        logger.info("Start to execute the logic of the runner")
        runner.run()
        logger.info("End of the run")


@hydra.main(config_path=None, version_base=None)
def run_cli(config: DictConfig) -> None:
    r"""Defines the CLI entrypoint to run an experiment.

    Please check the Hydra dcoumentation to learn how Hydra works:
    https://hydra.cc/

    Args:
        config (``omegaconf.DictConfig``): Specifies the dictionary
            with the configuration of the runner. This dictionary has
            to have a key ``'runner'``.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.cli.run import run_cli
        >>> run_cli()
    """
    log_run_info(config)
    register_resolvers()
    main(OmegaConf.to_container(config, resolve=True))


if __name__ == "__main__":
    run_cli()
