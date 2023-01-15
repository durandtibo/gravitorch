__all__ = ["LeNet5"]

from collections import OrderedDict

import torch
from torch import Tensor, nn

from gravitorch import constants as ct
from gravitorch.nn.utils.module_helpers import get_module_device


class LeNet5(nn.Module):
    r"""Implementation of LeNet5 network architecture.

    The main target of this network is MNIST.

    Args:
        num_classes (int): Specifies the number of classes.
        logits (bool, optional): Specifies if the output is the logits
            or the probabilities. By default, this network returns the
            logits. Default: ``True``
    """

    def __init__(
        self,
        num_classes: int,
        logits: bool = True,
        input_name: str = ct.INPUT,
        output_name: str = ct.PREDICTION,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
                    ),
                    ("tanh1", nn.Tanh()),
                    ("pool1", nn.AvgPool2d(kernel_size=2)),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)
                        ),
                    ),
                    ("tanh2", nn.Tanh()),
                    ("pool2", nn.AvgPool2d(kernel_size=2)),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1)
                        ),
                    ),
                    ("tanh3", nn.Tanh()),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("linear4", nn.Linear(in_features=120, out_features=84)),
                    ("tanh4", nn.Tanh()),
                    ("linear5", nn.Linear(in_features=84, out_features=num_classes)),
                ]
            )
        )
        if not logits:
            self.classifier.add_module("softmax", nn.Softmax(dim=1))

        self._input_name = str(input_name)
        self._output_name = str(output_name)

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Computes the predictions of the LeNet network.

        Args:
            tensor (``torch.Tensor`` of shape
                ``(batch size, 1, 32, 32)`` and type float): Specifies
                the input images.

        Returns:
            ``torch.Tensor`` of shape ``(batch size, num classes)``
                and type float: The logits or the probabilities for
                each class.
        """
        x = self.feature_extractor(tensor)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_dummy_input(self, batch_size: int = 1) -> tuple[Tensor]:
        r"""Generates a dummy input for the MLP.

        Args:
            batch_size (int, optional): Specifies the batch size to
                use to generate the dummy input. Default: ``1``

        Returns:
            ``tuple[Tensor]``: A tuple with one tensor of shape
                ``(batch_size, 1, 32, 32)`` and type ``float``.
                The tensor is on the same device that this network.
        """
        device = get_module_device(self.feature_extractor)
        return (torch.randn(batch_size, 1, 32, 32, device=device),)

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

        The order of the name should be the same that the order in the
        outputs of the forward function.

        Returns:
            tuple: The tuple of output names.
        """
        return (self._output_name,)
