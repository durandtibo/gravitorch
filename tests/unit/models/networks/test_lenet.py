import torch
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.models import VanillaModel
from gravitorch.models.criterions import VanillaLoss
from gravitorch.models.networks.lenet import LeNet5
from gravitorch.models.utils import is_loss_decreasing_with_sgd
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


############################
#     Tests for LeNet5     #
############################


@mark.parametrize("num_classes", SIZES)
def test_lenet5_num_classes(num_classes: int):
    assert LeNet5(num_classes).classifier.linear5.out_features == num_classes


def test_lenet5_with_softmax():
    assert isinstance(LeNet5(num_classes=10, logits=False).classifier.softmax, nn.Softmax)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_lenet5_forward(device: str, batch_size: int, mode: bool):
    device = torch.device(device)
    network = LeNet5(4).to(device=device)
    network.train(mode)
    out = network(torch.zeros(batch_size, 1, 32, 32, device=device))
    assert out.shape == (batch_size, 4)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", (True, False))
def test_lenet5_forward_without_logits(device: str, mode: bool):
    device = torch.device(device)
    network = LeNet5(3, logits=False).to(device=device)
    network.train(mode)
    assert (
        network(torch.zeros(2, 1, 32, 32, device=device))
        .sum(dim=1)
        .allclose(
            torch.ones(2, device=device),
            atol=1e-6,
        )
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_lenet5_get_dummy_input(device: str, batch_size: int):
    device = torch.device(device)
    network = LeNet5(num_classes=10).to(device=device)
    dummy_input = network.get_dummy_input(batch_size)
    assert len(dummy_input) == 1
    assert dummy_input[0].shape == (batch_size, 1, 32, 32)
    assert dummy_input[0].dtype == torch.float
    assert dummy_input[0].device == device


@mark.parametrize("input_name", ("image", ct.INPUT))
def test_lenet5_get_input_names(input_name: str):
    assert LeNet5(num_classes=10, input_name=input_name).get_input_names() == (input_name,)


@mark.parametrize("output_name", ("output", ct.PREDICTION))
def test_lenet5_get_output_names(output_name: str):
    assert LeNet5(num_classes=10, output_name=output_name).get_output_names() == (output_name,)


@mark.parametrize("input_name", ("image", ct.INPUT))
@mark.parametrize("output_name", ("output", ct.PREDICTION))
def test_lenet5_get_onnx_dynamic_axis(input_name: str, output_name: str):
    assert LeNet5(
        num_classes=10, input_name=input_name, output_name=output_name
    ).get_onnx_dynamic_axis() == {
        input_name: {0: "batch"},
        output_name: {0: "batch"},
    }


def test_lenet5_get_onnx_dynamic_axis_default():
    assert LeNet5(num_classes=10).get_onnx_dynamic_axis() == {
        ct.INPUT: {0: "batch"},
        ct.PREDICTION: {0: "batch"},
    }


def test_lenet5_is_loss_decreasing():
    assert is_loss_decreasing_with_sgd(
        model=VanillaModel(
            network=LeNet5(num_classes=10),
            criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        ),
        batch={ct.INPUT: torch.zeros(2, 1, 32, 32), ct.TARGET: torch.tensor([1, 2])},
    )
