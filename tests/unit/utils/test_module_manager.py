from collections import OrderedDict

import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import nn

from gravitorch import constants as ct
from gravitorch.utils.module_manager import ModuleManager

NAMES = ("NAME", "my_module")


###################################
#     Tests for ModuleManager     #
###################################


def test_module_manager_str_empty() -> None:
    assert str(ModuleManager()) == "ModuleManager()"


def test_module_manager_str_with_module() -> None:
    manager = ModuleManager()
    manager.add_module("my_module", nn.Linear(4, 5))
    assert str(manager).startswith("ModuleManager(")


def test_module_manager_len_empty() -> None:
    assert len(ModuleManager()) == 0


def test_module_manager_len_1_module() -> None:
    manager = ModuleManager()
    manager.add_module("my_module1", nn.Linear(4, 5))
    assert len(manager) == 1


def test_module_manager_len_2_modules() -> None:
    manager = ModuleManager()
    manager.add_module("my_module1", nn.Linear(4, 5))
    manager.add_module("my_module2", nn.Linear(4, 5))
    assert len(manager) == 2


@mark.parametrize("name", NAMES)
def test_module_manager_add_module(name: str) -> None:
    manager = ModuleManager()
    manager.add_module(name, nn.Linear(4, 5))
    assert name in manager._modules


def test_module_manager_add_module_override() -> None:
    manager = ModuleManager()
    manager.add_module("my_module", 1)
    manager.add_module("my_module", 2)
    assert manager._modules["my_module"] == 2


def test_module_manager_get_module() -> None:
    manager = ModuleManager()
    manager.add_module("my_module", nn.Linear(4, 5))
    assert isinstance(manager.get_module("my_module"), nn.Linear)


def test_module_manager_get_module_missing() -> None:
    manager = ModuleManager()
    with raises(ValueError, match="The module 'my_module' does not exist"):
        manager.get_module("my_module")


def test_module_manager_has_module_true() -> None:
    manager = ModuleManager()
    manager.add_module("my_module", nn.Linear(4, 5))
    assert manager.has_module("my_module")


def test_module_manager_has_module_false() -> None:
    assert not ModuleManager().has_module("my_module")


def test_module_manager_remove_module_exists() -> None:
    manager = ModuleManager()
    manager.add_module("my_module", nn.Linear(4, 5))
    manager.remove_module("my_module")
    assert len(manager._modules) == 0


def test_module_manager_remove_module_missing() -> None:
    manager = ModuleManager()
    with raises(
        ValueError,
        match="The module 'my_module' does not exist so it is not possible to remove it",
    ):
        manager.remove_module("my_module")


def test_module_manager_state_dict_1_module_with_state_dict_method() -> None:
    manager = ModuleManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(1.0)
    manager.add_module(ct.MODEL, model)
    state_dict = manager.state_dict()
    assert objects_are_equal(
        state_dict,
        {
            ct.MODEL: OrderedDict(
                [
                    ("weight", torch.ones(5, 4)),
                    ("bias", torch.ones(5)),
                ]
            )
        },
    )


def test_module_manager_state_dict_1_module_without_state_dict_method() -> None:
    manager = ModuleManager()
    manager.add_module(ct.MODEL, 12345)
    state_dict = manager.state_dict()
    assert state_dict == {}


def test_module_manager_state_dict_2_modules() -> None:
    manager = ModuleManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(1.0)
    manager.add_module(ct.MODEL, model)
    manager.add_module(ct.OPTIMIZER, torch.optim.SGD(model.parameters(), lr=0.01))
    state_dict = manager.state_dict()
    assert objects_are_equal(
        state_dict,
        {
            ct.MODEL: OrderedDict(
                [
                    ("weight", torch.ones(5, 4)),
                    ("bias", torch.ones(5)),
                ]
            ),
            ct.OPTIMIZER: {
                "state": {},
                "param_groups": [
                    {
                        "dampening": 0,
                        "foreach": None,
                        "lr": 0.01,
                        "maximize": False,
                        "momentum": 0,
                        "nesterov": False,
                        "params": [0, 1],
                        "weight_decay": 0,
                        "differentiable": False,
                    }
                ],
            },
        },
    )


def test_module_manager_load_state_dict_1_module() -> None:
    manager = ModuleManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(2.0)
    model.bias.data.fill_(2.0)
    manager.add_module(ct.MODEL, model)
    manager.load_state_dict(
        state_dict={
            ct.MODEL: {
                "weight": torch.ones(5, 4),
                "bias": torch.ones(5),
            }
        },
        keys=(ct.MODEL,),
    )
    assert model.weight.data.equal(torch.ones(5, 4))
    assert model.bias.data.equal(torch.ones(5))


def test_module_manager_load_state_dict_2_modules() -> None:
    manager = ModuleManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(2.0)
    model.bias.data.fill_(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    manager.add_module(ct.MODEL, model)
    manager.add_module(ct.OPTIMIZER, optimizer)
    manager.load_state_dict(
        state_dict={
            ct.MODEL: {
                "weight": torch.ones(5, 4),
                "bias": torch.ones(5),
            },
            ct.OPTIMIZER: {
                "state": {},
                "param_groups": [
                    {
                        "lr": 0.001,
                        "momentum": 0,
                        "dampening": 0,
                        "weight_decay": 0,
                        "nesterov": False,
                        "params": [0, 1],
                        "maximize": False,
                    }
                ],
            },
        },
    )
    assert model.weight.data.equal(torch.ones(5, 4))
    assert model.bias.data.equal(torch.ones(5))
    assert optimizer.param_groups[0]["lr"] == 0.001


def test_module_manager_load_state_dict_missing_key_in_state_dict() -> None:
    manager = ModuleManager()
    manager.load_state_dict(state_dict={}, keys=(ct.MODEL,))
    assert len(manager) == 0


def test_module_manager_load_state_dict_missing_module() -> None:
    manager = ModuleManager()
    manager.load_state_dict(state_dict={ct.MODEL: 123}, keys=(ct.MODEL,))
    assert len(manager) == 0


def test_module_manager_load_state_dict_no_load_state_dict() -> None:
    manager = ModuleManager()
    manager.add_module(ct.MODEL, 1)
    manager.load_state_dict(state_dict={ct.MODEL: 123}, keys=(ct.MODEL,))
    assert len(manager) == 1
