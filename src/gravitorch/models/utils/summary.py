r"""Defines some utilities function to analyze a ``torch.nn.Module``.

Inspired from https://github.com/PyTorchLightning/pytorch-
lightning/blob/master/pytorch_lightning/core/memory.py
"""

__all__ = ["ModelSummary"]

from collections import OrderedDict

import torch
from tabulate import tabulate
from torch.nn import Module

from gravitorch.nn.utils.summary import (
    ModuleSummary,
    multiline_format_dtype,
    multiline_format_size,
)
from gravitorch.utils.format import human_count


class ModelSummary:
    r"""Generates a summary of all layers in a ``torch.nn.Module``.

    Args:
        model (``torch.nn.Module``): The model to summarize.
        mode: Can be one of

             - `top` (default): only the top-level modules will be
                recorded (the children of the root module)
             - `full`: summarizes all layers and their submodules
                in the root module

    The string representation of this summary prints a table with
    columns containing the name, type and number of parameters for
    each layer.

    The root module may also have a function ``get_dummy_input`` as
    shown in the example below. If present, the root module will be
    called with it as input to determine the intermediate input and
    output shapes and data types of all layers. Supported are tensors
    and nested lists and tuples of tensors. All other types of inputs
    will be skipped and show as `?` in the summary table. The summary
    will also display `?` for layers not used in the forward pass.

    Example::

        >>> import torch
        >>> from torch import nn
        >>> from gravitorch.models.utils import ModelSummary
        >>> model = torch.nn.Linear(4, 5)
        >>> print(ModelSummary(model, mode='top'))  # doctest: +NORMALIZE_WHITESPACE
        ╒════╤═══════════════╤════════╤══════════╤════════════════╕
        │    │ Name          │ Type   │   Params │   Learn Params │
        ╞════╪═══════════════╪════════╪══════════╪════════════════╡
        │  0 │ [root module] │ Linear │     25   │           25   │
        ╘════╧═══════════════╧════════╧══════════╧════════════════╛
         - 25         Learnable params
         - 0          Non-learnable params
         - 25         Total params

        >>> from gravitorch.models.networks import BetaMLP
        >>> from gravitorch.nn import ConcatFusion
        >>> class MyNetwork(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fusion = ConcatFusion()
        ...         self.decoder = BetaMLP(input_size=20, hidden_sizes=(16, 7))
        ...
        ...     def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...         return self.decoder(self.fusion(x1, x2, x3)), x3
        ...
        ...     def get_dummy_input(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...         return (
        ...             torch.randn(batch_size, 10),
        ...             torch.randn(batch_size, 2),
        ...             torch.randn(batch_size, 8),
        ...         )
        ...
        >>> model = MyNetwork()
        >>> ModelSummary(model, mode='top')  # doctest: +NORMALIZE_WHITESPACE
        ╒════╤═══════════════╤══════════════╤══════════╤════════════════╤════════════╤═══════════════╤═════════════╤═══════════════╕  # noqa: E501,B950
        │    │ Name          │ Type         │   Params │   Learn Params │ In sizes   │ In dtype      │ Out sizes   │ Out dtype     │  # noqa: E501,B950
        ╞════╪═══════════════╪══════════════╪══════════╪════════════════╪════════════╪═══════════════╪═════════════╪═══════════════╡  # noqa: E501,B950
        │  0 │ [root module] │ MyNetwork    │    455   │          455   │ (1, 10)    │ torch.float32 │ (1, 7)      │ torch.float32 │  # noqa: E501,B950
        │    │               │              │          │                │ (1, 2)     │ torch.float32 │ (1, 8)      │ torch.float32 │  # noqa: E501,B950
        │    │               │              │          │                │ (1, 8)     │ torch.float32 │             │               │  # noqa: E501,B950
        ├────┼───────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  1 │ fusion        │ ConcatFusion │      0   │            0   │ (1, 10)    │ torch.float32 │ (1, 20)     │ torch.float32 │  # noqa: E501,B950
        │    │               │              │          │                │ (1, 2)     │ torch.float32 │             │               │  # noqa: E501,B950
        │    │               │              │          │                │ (1, 8)     │ torch.float32 │             │               │  # noqa: E501,B950
        ├────┼───────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  2 │ decoder       │ BetaMLP      │    455   │          455   │ (1, 20)    │ torch.float32 │ (1, 7)      │ torch.float32 │  # noqa: E501,B950
        ╘════╧═══════════════╧══════════════╧══════════╧════════════════╧════════════╧═══════════════╧═════════════╧═══════════════╛  # noqa: E501,B950
         - 455        Learnable params
         - 0          Non-learnable params
         - 455        Total params

        >>> ModelSummary(model, mode='full')  # doctest: +NORMALIZE_WHITESPACE
        ╒════╤════════════════════════╤══════════════╤══════════╤════════════════╤════════════╤═══════════════╤═════════════╤═══════════════╕  # noqa: E501,B950
        │    │ Name                   │ Type         │   Params │   Learn Params │ In sizes   │ In dtype      │ Out sizes   │ Out dtype     │  # noqa: E501,B950
        ╞════╪════════════════════════╪══════════════╪══════════╪════════════════╪════════════╪═══════════════╪═════════════╪═══════════════╡  # noqa: E501,B950
        │  0 │ [root module]          │ MyNetwork    │    455   │          455   │ (1, 10)    │ torch.float32 │ (1, 7)      │ torch.float32 │  # noqa: E501,B950
        │    │                        │              │          │                │ (1, 2)     │ torch.float32 │ (1, 8)      │ torch.float32 │  # noqa: E501,B950
        │    │                        │              │          │                │ (1, 8)     │ torch.float32 │             │               │  # noqa: E501,B950
        ├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  1 │ fusion                 │ ConcatFusion │      0   │            0   │ (1, 10)    │ torch.float32 │ (1, 20)     │ torch.float32 │  # noqa: E501,B950
        │    │                        │              │          │                │ (1, 2)     │ torch.float32 │             │               │  # noqa: E501,B950
        │    │                        │              │          │                │ (1, 8)     │ torch.float32 │             │               │  # noqa: E501,B950
        ├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  2 │ decoder                │ BetaMLP      │    455   │          455   │ (1, 20)    │ torch.float32 │ (1, 7)      │ torch.float32 │  # noqa: E501,B950
        ├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  3 │ decoder.layers         │ Sequential   │    455   │          455   │ (1, 20)    │ torch.float32 │ (1, 7)      │ torch.float32 │  # noqa: E501,B950
        ├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  4 │ decoder.layers.linear1 │ Linear       │    336   │          336   │ (1, 20)    │ torch.float32 │ (1, 16)     │ torch.float32 │  # noqa: E501,B950
        ├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  5 │ decoder.layers.relu1   │ ReLU         │      0   │            0   │ (1, 16)    │ torch.float32 │ (1, 16)     │ torch.float32 │  # noqa: E501,B950
        ├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤  # noqa: E501,B950
        │  6 │ decoder.layers.linear2 │ Linear       │    119   │          119   │ (1, 16)    │ torch.float32 │ (1, 7)      │ torch.float32 │  # noqa: E501,B950
        ╘════╧════════════════════════╧══════════════╧══════════╧════════════════╧════════════╧═══════════════╧═════════════╧═══════════════╛  # noqa: E501,B950
         - 455        Learnable params
         - 0          Non-learnable params
         - 455        Total params
    """

    MODE_TOP = "top"
    MODE_FULL = "full"
    MODE_DEFAULT = MODE_TOP
    MODES = [MODE_FULL, MODE_TOP]

    def __init__(self, model: Module, mode: str = MODE_DEFAULT):
        self._model = model
        self._mode = mode
        self._layer_summary = self.summarize()

    @property
    def named_modules(self) -> list[tuple[str, Module]]:
        modules = [("[root module]", self._model)]
        if self._mode == ModelSummary.MODE_FULL:
            # All the modules including the root module.
            modules.extend(list(self._model.named_modules())[1:])
            return modules
        elif self._mode == ModelSummary.MODE_TOP:
            # The children are the top-level modules.
            modules.extend(list(self._model.named_children()))
            return modules
        raise ValueError(f"Incorrect mode: {self._mode}. The valid modes are {self.MODES}.")

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self._layer_summary.keys())

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(layer.layer_type for layer in self._layer_summary.values())

    @property
    def in_sizes(self) -> tuple:
        return tuple(layer.in_size for layer in self._layer_summary.values())

    @property
    def out_sizes(self) -> tuple:
        return tuple(layer.out_size for layer in self._layer_summary.values())

    @property
    def in_dtypes(self) -> tuple:
        return tuple(layer.in_dtype for layer in self._layer_summary.values())

    @property
    def out_dtypes(self) -> tuple:
        return tuple(layer.out_dtype for layer in self._layer_summary.values())

    @property
    def param_nums(self) -> tuple[int, ...]:
        return tuple(layer.num_parameters for layer in self._layer_summary.values())

    @property
    def learn_param_nums(self) -> tuple[int, ...]:
        return tuple(layer.num_learnable_parameters for layer in self._layer_summary.values())

    def summarize(self) -> dict[str, ModuleSummary]:
        summary = OrderedDict((name, ModuleSummary(module)) for name, module in self.named_modules)
        model_forward_dummy_input(self._model)
        for layer in summary.values():
            layer.detach_hook()
        return summary

    def _create_table(self, tablefmt: str = "fancy_grid") -> str:
        tab = {
            "Name": self.layer_names,
            "Type": self.layer_types,
            "Params": list(map(human_count, self.param_nums)),
            "Learn Params": list(map(human_count, self.learn_param_nums)),
        }

        if hasattr(self._model, "get_dummy_input"):
            tab["In sizes"] = multiline_format_size(self.in_sizes)
            tab["In dtype"] = multiline_format_dtype(self.in_dtypes)
            tab["Out sizes"] = multiline_format_size(self.out_sizes)
            tab["Out dtype"] = multiline_format_dtype(self.out_dtypes)

        return tabulate(tab, headers="keys", tablefmt=tablefmt, showindex="always")

    def __str__(self) -> str:
        summary = self._create_table()

        learnable_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in self._model.parameters())
        non_learnable_parameters = total_parameters - learnable_parameters

        summary += f"\n - {human_count(learnable_parameters):<{10}} Learnable params\n"
        summary += f" - {human_count(non_learnable_parameters):<{10}} " f"Non-learnable params\n"
        summary += f" - {human_count(total_parameters):<{10}} Total params\n"
        return summary


def model_forward_dummy_input(model: Module) -> None:
    r"""Runs the example input through each layer to get input- and output sizes
    and data types.

    Args:
        model (``torch.nn.Module``): Specifies the model. The model
            should have the ``get_dummy_input`` method. This function
            is a noop if the model does not have the
            ``get_dummy_input`` method.
    """
    if not hasattr(model, "get_dummy_input"):
        return
    dummy_input = model.get_dummy_input()
    mode = model.training
    model.eval()
    with torch.no_grad():
        # Let the model hooks collect the input and output shapes and data types.
        if torch.is_tensor(dummy_input):
            model(dummy_input)
        else:
            model(*dummy_input)
    model.train(mode)  # restore mode of module
