r"""This module contains some tools to analyze the parameters of a
``torch.nn.Module``."""

from __future__ import annotations

__all__ = [
    "ParameterSummary",
    "get_parameter_summaries",
    "show_parameter_summary",
]

import logging
from dataclasses import asdict, dataclass

import torch
from tabulate import tabulate
from torch.nn import Module, Parameter, UninitializedParameter

from gravitorch.utils.mapping import convert_to_dict_of_lists

logger = logging.getLogger(__name__)


@dataclass
class ParameterSummary:
    r"""Implements a class to easily manage parameter summaries.

    NI: Not Initialized
    NP: No Parameter
    """
    name: str
    mean: float | str
    median: float | str
    std: float | str
    min: float | str
    max: float | str
    shape: tuple[int, ...] | str
    learnable: bool | str
    device: torch.device | str

    @classmethod
    def from_parameter(
        cls, name: str, parameter: Parameter | UninitializedParameter
    ) -> ParameterSummary:
        r"""Creates the parameter summary from the parameter object.

        Args:
        ----
            name (str): Specifies the name of the parameter.
            parameter (``torch.nn.Parameter`` or
                ``torch.nn.UninitializedParameter``): Specifies the
                parameter object.

        Returns:
        -------
            ``ParameterSummary``: The parameter summary.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from torch.nn import Parameter
            >>> from gravitorch.nn import ParameterSummary
            >>> ParameterSummary.from_parameter("weight", Parameter(torch.randn(6, 4)))
            ParameterSummary(name='weight', mean=..., median=..., std=..., min=..., max=..., shape=(6, 4), learnable=True, device=device(type='cpu'))
        """
        if isinstance(parameter, UninitializedParameter):
            return cls(
                name=name,
                mean="NI",
                median="NI",
                std="NI",
                min="NI",
                max="NI",
                shape="NI",
                learnable=parameter.requires_grad,
                device=parameter.device,
            )
        if parameter.numel() == 0:
            return cls(
                name=name,
                mean="NP",
                median="NP",
                std="NP",
                min="NP",
                max="NP",
                shape=tuple(parameter.shape),
                learnable=parameter.requires_grad,
                device=parameter.device,
            )
        return cls(
            name=name,
            mean=parameter.mean().item(),
            median=parameter.median().item(),
            std=parameter.std(dim=None).item(),
            min=parameter.min().item(),
            max=parameter.max().item(),
            shape=tuple(parameter.shape),
            learnable=parameter.requires_grad,
            device=parameter.device,
        )


def get_parameter_summaries(module: Module) -> list[ParameterSummary]:
    r"""Gets the parameter summaries of a module.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module with the
            parameters to summarize.

    Returns:
    -------
        list: The list of parameter summaries.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.nn import get_parameter_summaries
        >>> get_parameter_summaries(torch.nn.Linear(4, 6))
        [ParameterSummary(name='weight', mean=..., median=..., std=..., min=..., max=..., shape=(6, 4), learnable=True, device=device(type='cpu')),
         ParameterSummary(name='bias', mean=..., median=..., std=..., min=..., max=..., shape=(6,), learnable=True, device=device(type='cpu'))]
    """
    return [
        ParameterSummary.from_parameter(name, parameter)
        for name, parameter in module.named_parameters()
    ]


def show_parameter_summary(
    module: Module, tablefmt: str = "fancy_outline", floatfmt: str = ".6f"
) -> None:
    r"""Shows a summary of the model parameters.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to analyze.
        tablefmt (str, optional): Specifies the table format.
            Default: ``'fancy_outline'``
        floatfmt (str, optional): Specifies the float format.
            Default: ``'.6f'``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.nn import show_parameter_summary
        >>> show_parameter_summary(torch.nn.Linear(4, 6))
    """
    summaries = convert_to_dict_of_lists(
        [asdict(summary) for summary in get_parameter_summaries(module)]
    )
    logger.info(
        "Parameter summary\n"
        f'{tabulate(summaries, headers="keys", tablefmt=tablefmt, floatfmt=floatfmt)}\n'
    )
