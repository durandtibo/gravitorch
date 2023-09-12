from pytest import mark, raises

from gravitorch.utils.asset import AssetExistsError, AssetManager, AssetNotFoundError

NAMES = ("name", "mean")


##################################
#     Tests for AssetManager     #
##################################


def test_asset_manager_repr() -> None:
    assert repr(AssetManager({"name": 5})).startswith("AssetManager(")


def test_asset_manager_repr_empty() -> None:
    assert repr(AssetManager()).startswith("AssetManager(")


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
