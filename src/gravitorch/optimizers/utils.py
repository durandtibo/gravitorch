r"""This module implements some utility functions for the optimizers."""

from __future__ import annotations

__all__ = [
    "get_learning_rate_per_group",
    "get_weight_decay_per_group",
    "log_optimizer_parameters_per_group",
    "show_optimizer_parameters_per_group",
]

import logging
from typing import TYPE_CHECKING

from tabulate import tabulate
from torch.optim import Optimizer

from gravitorch.utils.exp_trackers import Step, sanitize_metrics
from gravitorch.utils.format import to_flat_dict

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


def get_learning_rate_per_group(optimizer: Optimizer) -> dict[int, float]:
    r"""Gets the learning rates to an optimizer.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the
            optimizer.

    Returns:
        set: The set of learning rates.
    """
    return _get_parameter_per_group(optimizer, "lr")


def get_weight_decay_per_group(optimizer: Optimizer) -> dict[int, float]:
    r"""Gets the weight decay for each group of an optimizer.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the
            optimizer.

    Returns:
        set: The set of weight decays.
    """
    return _get_parameter_per_group(optimizer, "weight_decay")


def _get_parameter_per_group(optimizer: Optimizer, key: str) -> dict:
    parameters = {}
    for i, params in enumerate(optimizer.param_groups):
        if key in params:
            parameters[i] = params[key]
    return parameters


def log_optimizer_parameters_per_group(
    optimizer: Optimizer,
    engine: BaseEngine,
    step: Step | None = None,
    prefix: str = "",
) -> None:
    r"""Logs the optimizer parameters for each group.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the
            optimizer.
        engine (``BaseEngine``): Specifies the engine used to log the
            optimizer parameters.
        step (``Step``, optional): Specifies the step used to log the
            optimizer parameters.
        prefix (``Step``, optional): Specifies the prefix used to log
            the optimizer parameters.
    """
    parameters = {}
    for i, group in enumerate(optimizer.param_groups):
        for key in sorted(group.keys()):
            if key != "params":
                parameters[f"{prefix}optimizer.group{i}.{key}"] = group[key]
    engine.log_metrics(sanitize_metrics(to_flat_dict(parameters)), step=step)


def show_optimizer_parameters_per_group(optimizer: Optimizer, tablefmt: str = "fancy_grid") -> None:
    r"""Shows the optimizer parameters for each group.

    This function uses the ``tabulate`` package to log the results in
    a table.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the
            optimizer.
        tablefmt (str, optional): Specifies the table format to show
            the optimizer information. You can find the valid formats
            at https://pypi.org/project/tabulate/.
            Default: ``'fancy_grid'``
    """
    lines = []
    for i, group in enumerate(optimizer.param_groups):
        line_group = [f"Group {i}"]
        for key in sorted(group.keys()):
            if key != "params":
                line_group.append(group[key])
        lines.append(line_group)

    headers = [key for key in sorted(optimizer.param_groups[0].keys()) if key != "params"]
    logger.info(
        f"Optimizer: parameters per group\n{tabulate(lines, headers=headers, tablefmt=tablefmt)}"
    )
