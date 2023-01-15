from unittest.mock import Mock

import torch
from coola import objects_are_equal
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.models.networks import ImageClassificationNetwork
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


################################################
#     Tests for ImageClassificationNetwork     #
################################################


@mark.parametrize("batch_size", SIZES)
def test_image_classification_network_forward(batch_size: int):
    module = Mock(spec=nn.Module, return_value=torch.ones(batch_size, 6))
    network = ImageClassificationNetwork(module)
    batch = torch.randn(batch_size, 3, 224, 224)
    assert network(batch).equal(torch.ones(batch_size, 6))
    assert objects_are_equal(module.call_args.args, (batch,))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_image_classification_network_get_dummy_input(device, batch_size):
    device = torch.device(device)
    network = ImageClassificationNetwork(nn.Conv2d(3, 5, 1, 1)).to(device=device)
    dummy_input = network.get_dummy_input(batch_size)
    assert len(dummy_input) == 1
    assert dummy_input[0].shape == (batch_size, 3, 224, 224)
    assert dummy_input[0].dtype == torch.float
    assert dummy_input[0].device == device


@mark.parametrize("input_name", ("image", ct.INPUT))
def test_image_classification_network_get_input_names(input_name: str):
    assert ImageClassificationNetwork(
        Mock(spec=nn.Module), input_name=input_name
    ).get_input_names() == (input_name,)


@mark.parametrize("output_name", ("output", ct.PREDICTION))
def test_image_classification_network_get_output_names(output_name: str):
    assert ImageClassificationNetwork(
        Mock(spec=nn.Module), output_name=output_name
    ).get_output_names() == (output_name,)


@mark.parametrize("input_name", ("image", ct.INPUT))
@mark.parametrize("output_name", ("output", ct.PREDICTION))
def test_image_classification_network_get_onnx_dynamic_axis(input_name: str, output_name: str):
    assert ImageClassificationNetwork(
        Mock(spec=nn.Module), input_name=input_name, output_name=output_name
    ).get_onnx_dynamic_axis() == {input_name: {0: "batch"}, output_name: {0: "batch"}}


def test_image_classification_network_get_onnx_dynamic_axis_default():
    assert ImageClassificationNetwork(Mock(spec=nn.Module)).get_onnx_dynamic_axis() == {
        ct.INPUT: {0: "batch"},
        ct.PREDICTION: {0: "batch"},
    }
