from unittest.mock import Mock

from objectory import OBJECT_TARGET

from gravitorch.experimental.rsrc import (
    BaseResource,
    PyTorchCudaBackend,
    setup_resource,
)

####################################
#     Tests for setup_resource     #
####################################


def test_setup_resource_object():
    runner = Mock(spec=BaseResource)
    assert setup_resource(runner) is runner


def test_setup_resource_dict():
    assert isinstance(
        setup_resource({OBJECT_TARGET: "gravitorch.experimental.rsrc.PyTorchCudaBackend"}),
        PyTorchCudaBackend,
    )
