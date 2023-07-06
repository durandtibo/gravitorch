from typing import Union

from objectory import OBJECT_TARGET
from pytest import mark
from torch.nn import Module, ReLU

from gravitorch.utils import setup_object

##################################
#     Tests for setup_object     #
##################################


@mark.parametrize("module", (ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}))
def test_setup_object(module: Union[Module, dict]) -> None:
    assert isinstance(setup_object(module), ReLU)


def test_setup_object_object() -> None:
    module = ReLU()
    assert setup_object(module) is module
