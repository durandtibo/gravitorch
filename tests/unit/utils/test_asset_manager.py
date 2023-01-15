from pytest import mark, raises

from gravitorch.utils import AssetManager
from gravitorch.utils.asset_manager import AssetNotFoundError

NAMES = ("name", "mean")


##################################
#     Tests for AssetManager     #
##################################


def test_asset_manager_str():
    assert str(AssetManager()).startswith("AssetManager()")


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


def test_asset_manager_get_asset_exists():
    manager = AssetManager()
    manager.add_asset("name", 5)
    assert manager.get_asset("name") == 5


def test_asset_manager_get_asset_does_not_exist():
    manager = AssetManager()
    with raises(AssetNotFoundError):
        manager.get_asset("name")


def test_asset_manager_has_asset_true():
    manager = AssetManager()
    manager.add_asset("name", 5)
    assert manager.has_asset("name")


def test_asset_manager_has_asset_false():
    assert not AssetManager().has_asset("name")


def test_asset_manager_remove_asset_exists():
    manager = AssetManager()
    manager.add_asset("name", 5)
    manager.remove_asset("name")
    assert not manager.has_asset("name")


def test_asset_manager_remove_asset_does_not_exist():
    manager = AssetManager()
    with raises(AssetNotFoundError):
        manager.remove_asset("name")
