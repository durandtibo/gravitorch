from unittest.mock import Mock

from objectory import OBJECT_TARGET

from gravitorch.experimental.rsrc import (
    BaseResourceManager,
    PyTorchCudaBackend,
    setup_resource_manager,
)

############################################
#     Tests for setup_resource_manager     #
############################################


def test_setup_resource_manager_object():
    runner = Mock(spec=BaseResourceManager)
    assert setup_resource_manager(runner) is runner


def test_setup_resource_manager_dict():
    assert isinstance(
        setup_resource_manager({OBJECT_TARGET: "gravitorch.experimental.rsrc.PyTorchCudaBackend"}),
        PyTorchCudaBackend,
    )
