from collections import OrderedDict

import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import nn

from gravitorch import constants as ct
from gravitorch.utils.asset import AssetExistsError, AssetManager, AssetNotFoundError

NAMES = ("name", "mean")


##################################
#     Tests for AssetManager     #
##################################


def test_asset_manager_len_empty() -> None:
    assert len(AssetManager()) == 0


def test_asset_manager_len_1_asset() -> None:
    manager = AssetManager()
    manager.add_asset("my_asset1", nn.Linear(4, 5))
    assert len(manager) == 1


def test_asset_manager_len_2_assets() -> None:
    manager = AssetManager()
    manager.add_asset("my_asset1", nn.Linear(4, 5))
    manager.add_asset("my_asset2", nn.Linear(4, 5))
    assert len(manager) == 2


def test_asset_manager_repr() -> None:
    assert repr(AssetManager({"name": 5})).startswith("AssetManager(")


def test_asset_manager_repr_empty() -> None:
    assert repr(AssetManager()) == "AssetManager()"


def test_asset_manager_str() -> None:
    assert str(AssetManager({"name": 5})) == "AssetManager(num_assets=1)"


def test_asset_manager_str_empty() -> None:
    assert str(AssetManager()) == "AssetManager(num_assets=0)"


@mark.parametrize("name", NAMES)
def test_asset_manager_add_asset(name: str) -> None:
    manager = AssetManager()
    manager.add_asset(name, 5)
    assert manager._assets == {name: 5}


def test_asset_manager_add_asset_duplicate_name_replace_ok_false() -> None:
    manager = AssetManager()
    manager.add_asset("name", 5)
    with raises(AssetExistsError, match="`name` is already used to register an asset."):
        manager.add_asset("name", 2)


def test_asset_manager_add_asset_duplicate_name_replace_ok_true() -> None:
    manager = AssetManager()
    manager.add_asset("name", 5)
    manager.add_asset("name", 2, replace_ok=True)
    assert manager._assets == {"name": 2}


def test_asset_manager_add_asset_multiple_assets() -> None:
    manager = AssetManager()
    manager.add_asset("name1", 5)
    manager.add_asset("name2", 2)
    assert manager._assets == {"name1": 5, "name2": 2}


def test_asset_manager_clone() -> None:
    manager = AssetManager({"name": 5})
    clone = manager.clone()
    manager.add_asset("name", 7, replace_ok=True)
    assert manager.equal(AssetManager({"name": 7}))
    assert clone.equal(AssetManager({"name": 5}))


def test_asset_manager_equal_true() -> None:
    assert AssetManager({"name": 5}).equal(AssetManager({"name": 5}))


def test_asset_manager_equal_true_empty() -> None:
    assert AssetManager().equal(AssetManager())


def test_asset_manager_equal_false_different_names() -> None:
    assert not AssetManager({"name1": 5}).equal(AssetManager({"name2": 5}))


def test_asset_manager_equal_false_different_values() -> None:
    assert not AssetManager({"name": 5}).equal(AssetManager({"name": 6}))


def test_asset_manager_equal_false_different_types() -> None:
    assert not AssetManager().equal("abc")


def test_asset_manager_get_asset_exists() -> None:
    assert AssetManager({"name": 5}).get_asset("name") == 5


def test_asset_manager_get_asset_does_not_exist() -> None:
    manager = AssetManager()
    with raises(AssetNotFoundError, match="The asset 'name' does not exist"):
        manager.get_asset("name")


def test_asset_manager_get_asset_names() -> None:
    assert AssetManager({"mean": 5, "std": 1.2}).get_asset_names() == ("mean", "std")


def test_asset_manager_get_asset_names_empty() -> None:
    assert AssetManager().get_asset_names() == ()


def test_asset_manager_has_asset_true() -> None:
    assert AssetManager({"name": 5}).has_asset("name")


def test_asset_manager_has_asset_false() -> None:
    assert not AssetManager().has_asset("name")


def test_asset_manager_remove_asset_exists() -> None:
    manager = AssetManager({"name": 5})
    manager.remove_asset("name")
    assert not manager.has_asset("name")


def test_asset_manager_remove_asset_does_not_exist() -> None:
    manager = AssetManager()
    with raises(
        AssetNotFoundError,
        match="The asset 'name' does not exist so it is not possible to remove it",
    ):
        manager.remove_asset("name")


def test_asset_manager_state_dict_1_asset_with_state_dict_method() -> None:
    manager = AssetManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(1.0)
    manager.add_asset(ct.MODEL, model)
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


def test_asset_manager_state_dict_1_asset_without_state_dict_method() -> None:
    manager = AssetManager()
    manager.add_asset(ct.MODEL, 12345)
    state_dict = manager.state_dict()
    assert state_dict == {}


def test_asset_manager_state_dict_2_assets() -> None:
    manager = AssetManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(1.0)
    manager.add_asset(ct.MODEL, model)
    manager.add_asset(ct.OPTIMIZER, torch.optim.SGD(model.parameters(), lr=0.01))
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


def test_asset_manager_load_state_dict_1_asset() -> None:
    manager = AssetManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(2.0)
    model.bias.data.fill_(2.0)
    manager.add_asset(ct.MODEL, model)
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


def test_asset_manager_load_state_dict_2_assets() -> None:
    manager = AssetManager()
    model = nn.Linear(4, 5)
    model.weight.data.fill_(2.0)
    model.bias.data.fill_(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    manager.add_asset(ct.MODEL, model)
    manager.add_asset(ct.OPTIMIZER, optimizer)
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


def test_asset_manager_load_state_dict_missing_key_in_state_dict() -> None:
    manager = AssetManager()
    manager.load_state_dict(state_dict={}, keys=(ct.MODEL,))
    assert len(manager) == 0


def test_asset_manager_load_state_dict_missing_asset() -> None:
    manager = AssetManager()
    manager.load_state_dict(state_dict={ct.MODEL: 123}, keys=(ct.MODEL,))
    assert len(manager) == 0


def test_asset_manager_load_state_dict_no_load_state_dict() -> None:
    manager = AssetManager()
    manager.add_asset(ct.MODEL, 1)
    manager.load_state_dict(state_dict={ct.MODEL: 123}, keys=(ct.MODEL,))
    assert len(manager) == 1
