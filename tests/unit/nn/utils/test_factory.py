from typing import Union

from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn
from torch.nn import ReLU

from gravitorch.nn import setup_module

#####################################
#     Tests for setup_module     #
#####################################


@mark.parametrize("module", (ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}))
def test_setup_module(module: Union[nn.Module, dict]):
    assert isinstance(setup_module(module), ReLU)


def test_setup_module_object():
    module = ReLU()
    assert setup_module(module) is module
