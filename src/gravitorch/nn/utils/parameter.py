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
            name (str): Specifies the name of the parameter.
            parameter (``torch.nn.Parameter`` or
                ``torch.nn.UninitializedParameter``): Specifies the
                parameter object.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from torch.nn import Parameter
            >>> from gravitorch.nn import ParameterSummary
            >>> ParameterSummary.from_parameter("weight", Parameter(torch.randn(6, 4)))
            ParameterSummary(name='weight', mean=0.10939589142799377, median=-0.018833406269550323,
            std=0.8310111165046692, min=-1.6590238809585571, max=2.0370750427246094,
            learnable=True, shape=(6, 4))
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
        module (``torch.nn.Module``): Specifies the module with the
            parameters to summarize.

    Returns:
        list: The list of parameter summaries.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from torch.nn import Linear
        >>> from gravitorch.nn import get_parameter_summaries
        >>> get_parameter_summaries(Linear(4, 6))
        [ParameterSummary(name='weight', mean=0.10450785607099533, median=0.02029263973236084,
          std=0.26641708612442017, min=-0.4861736297607422, max=0.48399144411087036,
          learnable=True, shape=(6, 4)),
         ParameterSummary(name='bias', mean=0.11823087930679321, median=-0.05595135688781738,
          std=0.36556652188301086, min=-0.4497765898704529, max=0.4593251347541809,
          learnable=True, shape=(6,))]
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
    """
    summaries = convert_to_dict_of_lists(
        [asdict(summary) for summary in get_parameter_summaries(module)]
    )
    logger.info(
        "Parameter summary\n"
        f'{tabulate(summaries, headers="keys", tablefmt=tablefmt, floatfmt=floatfmt)}\n'
    )
