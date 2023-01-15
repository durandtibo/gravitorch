r"""This module contains some tools to analyze the parameters of a
``torch.nn.Module``."""

__all__ = ["compute_parameter_stats", "show_parameter_stats"]

import logging

from tabulate import tabulate
from torch.nn import Module

logger = logging.getLogger(__name__)


def compute_parameter_stats(module: Module) -> list[list]:
    r"""Computes the parameter statistics of a ``torch.nn.Module``.

    Args:
        module (``torch.nn.Module``): Specifies the module to analyze.

    Returns:
        list: The list of statistics per parameters.
    """
    stats = [["parameter", "mean", "median", "std", "min", "max", "learnable"]]
    for key, weight in module.named_parameters():
        weight = weight.flatten()
        if weight.numel() > 0:
            stats.append(
                [
                    key,
                    weight.mean().item(),
                    weight.median().item(),
                    weight.std(dim=0).item(),
                    weight.min().item(),
                    weight.max().item(),
                    weight.requires_grad,
                ]
            )
    return stats


def show_parameter_stats(module: Module, tablefmt: str = "rst") -> None:
    r"""Shows some statistics about the model parameters.

    Args:
        module (``torch.nn.Module``): Specifies the module to analyze.
        tablefmt (str, optional): Specifies the table format.
            Default: ``'rst'``
    """
    stats = compute_parameter_stats(module)
    logger.info(
        "Parameters statistics\n"
        f'{tabulate(stats, headers="firstrow", tablefmt=tablefmt, floatfmt=".6f")}\n'
    )
