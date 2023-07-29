import logging
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Identity

from gravitorch.rsrc import BaseResource, PyTorchCudaBackend, setup_resource

####################################
#     Tests for setup_resource     #
####################################


def test_setup_resource_object() -> None:
    runner = Mock(spec=BaseResource)
    assert setup_resource(runner) is runner


def test_setup_resource_dict() -> None:
    assert isinstance(
        setup_resource({OBJECT_TARGET: "gravitorch.rsrc.PyTorchCudaBackend"}),
        PyTorchCudaBackend,
    )


def test_setup_resource_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_resource({OBJECT_TARGET: "torch.nn.Identity"}), Identity)
        assert caplog.messages
