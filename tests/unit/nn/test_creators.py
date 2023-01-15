from collections.abc import Sequence

from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn

from gravitorch.nn import create_sequential

#######################################
#     Tests for create_sequential     #
#######################################


@mark.parametrize(
    "modules",
    (
        [{OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}, nn.ReLU()],
        ({OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}, nn.ReLU()),
    ),
)
def test_create_sequential(modules: Sequence):
    module = create_sequential(modules)
    assert isinstance(module, nn.Sequential)
    assert len(module) == 2
    assert isinstance(module[0], nn.Linear)
    assert isinstance(module[1], nn.ReLU)


def test_create_sequential_empty():
    module = create_sequential([])
    assert isinstance(module, nn.Sequential)
    assert len(module) == 0
