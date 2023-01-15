__all__ = ["PyTorchMnistNet"]

from collections import OrderedDict

import torch
from torch import Tensor, nn

from gravitorch import constants as ct
from gravitorch.nn.utils.module_helpers import get_module_device


class PyTorchMnistNet(nn.Module):
    r"""Implements the network architecture from PyTorch example.

    https://github.com/pytorch/examples/blob/master/mnist/main.py

    Args:
        num_classes (int, optional): Specifies the number of classes.
            Default: 10
        input_name (str, optional): Specifies the name of the input.
            Default: ``'input'``
        output_name (str, optional): Specifies the name of the output.
            Default: ``'prediction'``
    """

    def __init__(
        self, num_classes: int = 10, input_name: str = ct.INPUT, output_name: str = ct.PREDICTION
    ):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 32, 3, 1)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(32, 64, 3, 1)),
                    ("relu2", nn.ReLU()),
                    ("pool", nn.MaxPool2d(kernel_size=2)),
                    ("dropout1", nn.Dropout(0.25)),
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(9216, 128)),
                    ("relu3", nn.ReLU()),
                    ("dropout2", nn.Dropout(0.5)),
                    ("fc2", nn.Linear(128, int(num_classes))),
                ]
            )
        )
        self._input_name = str(input_name)
        self._output_name = str(output_name)

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Computes the predictions.

        Args:
            tensor (``torch.Tensor`` of shape
                ``(batch size, 1, 28, 28)`` and type float):
                Specifies the input images.

        Returns:
            ``torch.Tensor`` of shape ``(batch size, num classes)``
                and type float: The logits for each class.
        """
        return self.net(tensor)

    def get_dummy_input(self, batch_size: int = 1) -> tuple[Tensor]:
        r"""Generates a dummy input for the MLP.

        Args:
            batch_size (int, optional): Specifies the batch size to
                use to generate the dummy input. Default: ``1``

        Returns:
            ``tuple[Tensor]``: A tuple with one tensor of shape
                ``(batch_size, 1, 28, 28)`` and type ``float``.
                The tensor is on the same device that this network.
        """
        return (torch.rand(batch_size, 1, 28, 28, device=get_module_device(self.net)),)

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

        The order of the name should be the same that the order in the
        outputs of the forward function.

        Returns:
            tuple: The tuple of output names.
        """
        return (self._output_name,)
