from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import mark

from gravitorch import constants as ct
from gravitorch.creators.datasource import DataSourceCreator
from gravitorch.datasources import BaseDataSource
from gravitorch.engines import BaseEngine
from gravitorch.testing import DummyDataSource

#######################################
#     Tests for DataSourceCreator     #
#######################################


def test_datasource_creator_str() -> None:
    assert str(DataSourceCreator(config={})).startswith("DataSourceCreator(")


@mark.parametrize("attach_to_engine", (True, False))
def test_datasource_creator_attach_to_engine(attach_to_engine: bool) -> None:
    assert (
        DataSourceCreator(
            config={},
            attach_to_engine=attach_to_engine,
        )._attach_to_engine
        == attach_to_engine
    )


@mark.parametrize("add_module_to_engine", (True, False))
def test_datasource_creator_add_module_to_engine(add_module_to_engine: bool) -> None:
    assert (
        DataSourceCreator(
            config={}, add_module_to_engine=add_module_to_engine
        )._add_module_to_engine
        == add_module_to_engine
    )


def test_datasource_creator_create() -> None:
    creator = DataSourceCreator(config={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"})
    datasource = creator.create(engine=Mock(spec=BaseEngine))
    assert isinstance(datasource, DummyDataSource)


def test_datasource_creator_create_attach_to_engine_true() -> None:
    creator = DataSourceCreator(config={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"})
    engine = Mock(spec=BaseEngine)
    datasource = Mock(spec=BaseDataSource)
    setup_mock = Mock(return_value=datasource)
    with patch("gravitorch.creators.datasource.vanilla.setup_and_attach_datasource", setup_mock):
        assert creator.create(engine) == datasource
        setup_mock.assert_called_once_with(
            datasource={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"}, engine=engine
        )


def test_datasource_creator_create_attach_to_engine_false() -> None:
    creator = DataSourceCreator(
        config={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"}, attach_to_engine=False
    )
    datasource = Mock(spec=BaseDataSource)
    setup_mock = Mock(return_value=datasource)
    with patch("gravitorch.creators.datasource.vanilla.setup_datasource", setup_mock):
        assert creator.create(engine=Mock(spec=BaseEngine)) == datasource
        setup_mock.assert_called_once_with(
            datasource={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"}
        )


def test_datasource_creator_create_add_module_to_engine_true() -> None:
    engine = Mock(spec=BaseEngine)
    creator = DataSourceCreator(config={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"})
    datasource = creator.create(engine)
    assert isinstance(datasource, DummyDataSource)
    engine.add_module.assert_called_once_with(ct.DATA_SOURCE, datasource)


def test_datasource_creator_create_add_module_to_engine_false() -> None:
    engine = Mock(spec=BaseEngine)
    creator = DataSourceCreator(
        config={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"},
        add_module_to_engine=False,
    )
    datasource = creator.create(engine)
    assert isinstance(datasource, DummyDataSource)
    engine.add_module.assert_not_called()
