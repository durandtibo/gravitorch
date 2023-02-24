r"""This module implements a wrapper network to make image classification
network from ``torchvision`` compatible with ``VanillaModel``."""

__all__ = ["ImageClassificationNetwork"]

from typing import Union

import torch
from torch import Tensor
from torch.nn import Module

from gravitorch import constants as ct
from gravitorch.nn.utils.factory import setup_module
from gravitorch.nn.utils.module_helpers import get_module_device


class ImageClassificationNetwork(Module):
    r"""Implements a wrapper network for image classification network from
    ``torchvision``.

    This network assumes that the input shape is ``224*224``.

    Args:
        network (``torch.nn.Module`` or dict): Specifies the network
            module or its configuration.
        input_name (str, optional): Specifies the input name.
            Default: ``'input'``
        output_name (str, optional): Specifies the output name.
            Default: ``'prediction'``
    """

    def __init__(
        self,
        network: Union[Module, dict],
        input_name: str = ct.INPUT,
        output_name: str = ct.PREDICTION,
    ):
        super().__init__()
        self.network = setup_module(network)
        self._input_name = input_name
        self._output_name = output_name

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Computes the predictions of the network.

        Args:
            tensor (``torch.Tensor`` of shape
                ``(batch size, num channels, height, width)`` and
                type float): Specifies the input images.

        Returns:
            ``torch.Tensor`` of shape ``(batch size, num classes)``
                and type float: The prediction for each class.
        """
        return self.network(tensor)

    def get_dummy_input(self, batch_size: int = 1) -> tuple[Tensor]:
        r"""Generates a dummy input for the MLP.

        Args:
            batch_size (int, optional): Specifies the batch size to
                use to generate the dummy input. Default: ``1``

        Returns:
            ``tuple[Tensor]``: A tuple with one tensor of shape
                ``(batch_size, 3, 244, 244)`` and type ``float``.
                The tensor is on the same device that this network.
        """
        return (torch.randn(batch_size, 3, 224, 224, device=get_module_device(self.network)),)

    def get_input_names(self) -> tuple[str]:
        r"""Gets the tuple of input names for the network.

        The order of the name should be the same that the order in
        the inputs of the forward function.

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
