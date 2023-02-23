import torch
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.models import VanillaModel
from gravitorch.models.criteria import VanillaLoss
from gravitorch.models.networks.mnist import PyTorchMnistNet
from gravitorch.models.utils import is_loss_decreasing_with_sgd
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


#####################################
#     Tests for PyTorchMnistNet     #
#####################################


@mark.parametrize("input_name", ("image", ct.INPUT))
def test_pytorch_mnist_net_get_input_names(input_name: str):
    assert PyTorchMnistNet(input_name=input_name).get_input_names() == (input_name,)


@mark.parametrize("output_name", ("output", ct.PREDICTION))
def test_pytorch_mnist_net_get_output_names(output_name: str):
    assert PyTorchMnistNet(output_name=output_name).get_output_names() == (output_name,)


@mark.parametrize("input_name", ("image", ct.INPUT))
@mark.parametrize("output_name", ("output", ct.PREDICTION))
def test_pytorch_mnist_net_get_onnx_dynamic_axis(input_name: str, output_name: str):
    assert PyTorchMnistNet(
        input_name=input_name, output_name=output_name
    ).get_onnx_dynamic_axis() == {
        input_name: {0: "batch"},
        output_name: {0: "batch"},
    }


def test_pytorch_mnist_net_get_onnx_dynamic_axis_default():
    assert PyTorchMnistNet().get_onnx_dynamic_axis() == {
        ct.INPUT: {0: "batch"},
        ct.PREDICTION: {0: "batch"},
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_pytorch_mnist_net_forward(device, batch_size, mode):
    device = torch.device(device)
    network = PyTorchMnistNet().to(device=device)
    network.train(mode)
    out = network(torch.randn(batch_size, 1, 28, 28, device=device))
    assert out.shape == (batch_size, 10)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_pytorch_mnist_net_get_dummy_input(device, batch_size):
    device = torch.device(device)
    dummy_input = PyTorchMnistNet().to(device=device).get_dummy_input(batch_size)
    assert len(dummy_input) == 1
    assert dummy_input[0].shape == (batch_size, 1, 28, 28)
    assert dummy_input[0].dtype == torch.float
    assert dummy_input[0].device == device


def test_pytorch_mnist_net_is_loss_decreasing():
    assert is_loss_decreasing_with_sgd(
        model=VanillaModel(
            network=PyTorchMnistNet(),
            criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        ),
        batch={ct.INPUT: torch.randn(2, 1, 28, 28), ct.TARGET: torch.tensor([1, 2])},
    )
