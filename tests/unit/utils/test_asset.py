import numpy as np
import torch
from pytest import mark, raises

from gravitorch.utils.asset import AssetManager, AssetNotFoundError, get_asset_summary

NAMES = ("name", "mean")


##################################
#     Tests for AssetManager     #
##################################


def test_asset_manager_repr():
    assert repr(AssetManager({"name": 5})).startswith("AssetManager(")


def test_asset_manager_repr_empty():
    assert repr(AssetManager()).startswith("AssetManager(")


def test_asset_manager_str():
    assert str(AssetManager({"name": 5})) == "AssetManager(num_assets=1)"


def test_asset_manager_str_empty():
    assert str(AssetManager()) == "AssetManager(num_assets=0)"


@mark.parametrize("name", NAMES)
def test_asset_manager_add_asset(name: str):
    manager = AssetManager()
    manager.add_asset(name, 5)
    assert manager._assets == {name: 5}


def test_asset_manager_add_asset_duplicate_name():
    manager = AssetManager()
    manager.add_asset("name", 5)
    manager.add_asset("name", 2)
    assert manager._assets == {"name": 2}


def test_asset_manager_add_asset_multiple_assets():
    manager = AssetManager()
    manager.add_asset("name1", 5)
    manager.add_asset("name2", 2)
    assert manager._assets == {"name1": 5, "name2": 2}


def test_asset_manager_clone():
    manager = AssetManager({"name": 5})
    clone = manager.clone()
    manager.add_asset("name", 7)
    assert manager.equal(AssetManager({"name": 7}))
    assert clone.equal(AssetManager({"name": 5}))


def test_asset_manager_equal_true():
    assert AssetManager({"name": 5}).equal(AssetManager({"name": 5}))


def test_asset_manager_equal_true_empty():
    assert AssetManager().equal(AssetManager())


def test_asset_manager_equal_false_different_names():
    assert not AssetManager({"name1": 5}).equal(AssetManager({"name2": 5}))


def test_asset_manager_equal_false_different_values():
    assert not AssetManager({"name": 5}).equal(AssetManager({"name": 6}))


def test_asset_manager_equal_false_different_types():
    assert not AssetManager().equal("abc")


def test_asset_manager_get_asset_exists():
    assert AssetManager({"name": 5}).get_asset("name") == 5


def test_asset_manager_get_asset_does_not_exist():
    manager = AssetManager()
    with raises(AssetNotFoundError):
        manager.get_asset("name")


def test_asset_manager_get_asset_names():
    assert AssetManager({"mean": 5, "std": 1.2}).get_asset_names() == ("mean", "std")


def test_asset_manager_get_asset_names_empty():
    assert AssetManager().get_asset_names() == tuple()


def test_asset_manager_has_asset_true():
    assert AssetManager({"name": 5}).has_asset("name")


def test_asset_manager_has_asset_false():
    assert not AssetManager().has_asset("name")


def test_asset_manager_remove_asset_exists():
    manager = AssetManager({"name": 5})
    manager.remove_asset("name")
    assert not manager.has_asset("name")


def test_asset_manager_remove_asset_does_not_exist():
    manager = AssetManager()
    with raises(AssetNotFoundError):
        manager.remove_asset("name")


#######################################
#     Tests for get_asset_summary     #
#######################################


def test_get_asset_summary_tensor():
    assert (
        get_asset_summary(torch.ones(2, 3))
        == "<class 'torch.Tensor'>  shape=torch.Size([2, 3])  dtype=torch.float32  device=cpu"
    )


def test_get_asset_summary_ndarray():
    assert (
        get_asset_summary(np.ones((2, 3))) == "<class 'numpy.ndarray'>  shape=(2, 3)  dtype=float64"
    )


def test_get_asset_summary_bool():
    assert get_asset_summary(True) == "<class 'bool'>  True"


def test_get_asset_summary_int():
    assert get_asset_summary(42) == "<class 'int'>  42"


def test_get_asset_summary_float():
    assert get_asset_summary(2.35) == "<class 'float'>  2.35"


def test_get_asset_summary_str():
    assert get_asset_summary("abc") == "<class 'str'>"
