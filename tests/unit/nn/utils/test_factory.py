from typing import Union

from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn
from torch.nn import ReLU

from gravitorch.nn import is_module_config, setup_module

######################################
#     Tests for is_module_config     #
######################################


def test_is_module_config_true() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


def test_is_module_config_false() -> None:
    assert not is_module_config({OBJECT_TARGET: "gravitorch.loops.training.VanillaTrainingLoop"})


##################################
#     Tests for setup_module     #
##################################


@mark.parametrize("module", (ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}))
def test_setup_module(module: Union[nn.Module, dict]) -> None:
    assert isinstance(setup_module(module), ReLU)


def test_setup_module_object() -> None:
    module = ReLU()
    assert setup_module(module) is module
