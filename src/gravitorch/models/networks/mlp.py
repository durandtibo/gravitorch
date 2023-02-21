__all__ = ["BaseMLP", "AlphaMLP", "BetaMLP", "create_alpha_mlp", "create_beta_mlp"]

from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional

import torch
from objectory import OBJECT_TARGET
from torch import Tensor, nn

from gravitorch import constants as ct
from gravitorch.nn.utils.factory import setup_module
from gravitorch.nn.utils.module_helpers import get_module_device, get_module_name


class BaseMLP(nn.Module):
    r"""Defines a base class to implement a MLP architecture.

    Args:
        input_size (int): Specifies the input size of the MLP network.
        input_name (str, optional): Specifies the name of the input.
            This name is used to identify the input in the batch and
            to export the network. Default: ``'input'``
        output_name (str, optional): Specifies the name of the output.
            This name is used to create the network output dictionary
            and to export the network. Default: ``'prediction'``
    """

    def __init__(
        self,
        input_size: int,
        input_name: str = ct.INPUT,
        output_name: str = ct.PREDICTION,
    ):
        super().__init__()
        self._input_size = input_size
        self._input_name = input_name
        self._output_name = output_name

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def input_size(self) -> int:
        r"""int: The input size of the MLP network."""
        return self._input_size

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Computes the predictions of the MLP network.

        Args:
            inputs (``torch.Tensor`` of shape
                ``(batch_size, input_size)`` and type ``float``):
                Specifies the input of the MLP network.

        Returns:
            ``torch.Tensor`` of shape ``(batch_size, output_size)``
                and type ``float``: The predictions.
        """
        return self.layers(inputs)

    def get_dummy_input(self, batch_size: int = 1) -> tuple[Tensor]:
        r"""Generates a dummy input for the MLP.

        Args:
            batch_size (int, optional): Specifies the batch size to
                use to generate the dummy input. Default: ``1``

        Returns:
            ``tuple[Tensor]``: A tuple with one tensor of shape
                ``(batch_size, input_size)`` and type ``float``.
                The tensor is on the same device that this network.
        """
        return (torch.randn(batch_size, self._input_size, device=get_module_device(self.layers)),)

    def get_input_names(self) -> tuple[str]:
        r"""Gets the tuple of input names for the network.

        The order of the name should be the same that the order in the
        inputs of the forward function.

        Returns:
            tuple: The tuple of input names.
        """
        return (self._input_name,)

    def get_onnx_dynamic_axis(self) -> dict:
        r"""Gets the dynamic axes (e.g. batch size or sequence length) when
        serializing a model to ONNX format.

        See https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
        to have more information on how to create the ``dict``.

        Returns:
            dict: with the dynamic axes of the input/output.
        """
        return {self._input_name: {0: "batch"}, self._output_name: {0: "batch"}}

    def get_output_names(self) -> tuple[str]:
        r"""Gets the tuple of output names for the network.

        The order of the name should be the same that the order in
        the outputs of the forward function.

        Returns:
            tuple: The tuple of output names.
        """
        return (self._output_name,)


class AlphaMLP(BaseMLP):
    r"""Implements a MLP network where the last layer is an activation layer.

    Args:
        input_size (int): Specifies the input size of the MLP network.
        hidden_sizes (sequence): Specifies the hidden sizes of the
            MLP network. The last size is the output size of the MLP.
            This input should have at least one value.
        activation (dict or ``None``, optional): Specifies the
            configuration the activation layer. Default: ``None``
        dropout (float, optional): Specifies the dropout probability
            of an element to be zeroed. Default: ``0.0``
        input_name (str, optional): Specifies the name of the input.
            This name is used to identify the input in the batch and
            to export the network. Default: ``'input'``
        output_name (str, optional): Specifies the name of the output.
            This name is used to create the network output
            dictionary and to export the network.
            Default: ``'prediction'``
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        activation: Optional[dict] = None,
        dropout: float = 0.0,
        input_name: str = ct.INPUT,
        output_name: str = ct.PREDICTION,
    ):
        super().__init__(input_size=input_size, input_name=input_name, output_name=output_name)
        self.layers = create_alpha_mlp(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
        )


class BetaMLP(BaseMLP):
    r"""Implements a MLP network where the last layer is a linear layer (without
    activation).

    Args:
        input_size (int): Specifies the input size of the MLP network.
        hidden_sizes (sequence): Specifies the hidden sizes of the
            MLP network. The last size is the output size of the MLP.
            This input should have at least one value.
        activation (dict or ``None``, optional): Specifies the
            configuration the activation layer. Default: ``None``
        dropout (float, optional): Specifies the dropout probability
            of an element to be zeroed. Default: ``0.0``
        input_name (str, optional): Specifies the name of the input.
            This name is used to identify the input in the batch and
            to export the network. Default: ``'input'``
        output_name (str, optional): Specifies the name of the output.
            This name is used to create the network output
            dictionary and to export the network.
            Default: ``'prediction'``
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        activation: Optional[dict] = None,
        dropout: float = 0.0,
        input_name: str = ct.INPUT,
        output_name: str = ct.PREDICTION,
    ):
        super().__init__(input_size=input_size, input_name=input_name, output_name=output_name)
        self.layers = create_beta_mlp(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
        )


def create_alpha_mlp(
    input_size: int,
    hidden_sizes: Sequence[int],
    activation: Optional[dict] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    r"""Creates a MLP network where the last layer is an activation layer.

    Args:
        input_size (int): Specifies the input size of the MLP network.
        hidden_sizes (sequence): Specifies the hidden sizes of the
            MLP network. The last size is the output size of the MLP.
            This input should have at least one value.
        activation (dict or ``None``, optional): Specifies the
            configuration the activation layer. Default: ``None``
        dropout (float, optional): Specifies the dropout probability
            of an element to be zeroed. Default: ``0.0``

    Returns:
        ``torch.nn.Sequential``: The instantiated MLP network.
    """
    activation = activation or {OBJECT_TARGET: "torch.nn.ReLU"}
    sizes = (input_size, *hidden_sizes)
    layers = OrderedDict()
    for i in range(len(sizes) - 1):
        if dropout > 0:
            layers[f"dropout{i + 1}"] = nn.Dropout(dropout)
        layers[f"linear{i + 1}"] = nn.Linear(sizes[i], sizes[i + 1])
        activation_layer = setup_module(activation)
        layers[f"{get_module_name(activation_layer).lower()}{i + 1}"] = activation_layer
    return nn.Sequential(layers)


def create_beta_mlp(
    input_size: int,
    hidden_sizes: Sequence[int],
    activation: Optional[dict] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    r"""Creates a MLP network where the last layer is a linear layer (without
    activation).

    Args:
        input_size (int): Specifies the input size of the MLP network.
        hidden_sizes (sequence): Specifies the hidden sizes of the
            MLP network. The last size is the output size of the MLP.
            This input should have at least one value.
        activation (dict or ``None``, optional): Specifies the
            configuration the activation layer. Default: ``None``
        dropout (float, optional): Specifies the dropout probability
            of an element to be zeroed. Default: ``0.0``

    Returns:
        ``torch.nn.Sequential``: The instantiated MLP network.
    """
    activation = activation or {OBJECT_TARGET: "torch.nn.ReLU"}
    sizes = (input_size, *hidden_sizes)
    layers = OrderedDict()
    for i in range(len(sizes) - 1):
        if dropout > 0:
            layers[f"dropout{i + 1}"] = nn.Dropout(dropout)
        layers[f"linear{i + 1}"] = nn.Linear(sizes[i], sizes[i + 1])
        activation_layer = setup_module(activation)
        if i < len(sizes) - 2:
            layers[f"{get_module_name(activation_layer).lower()}{i + 1}"] = activation_layer
    return nn.Sequential(layers)
