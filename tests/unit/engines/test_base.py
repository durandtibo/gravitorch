import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Identity

from gravitorch.engines import AlphaEngine, BaseEngine, is_engine_config, setup_engine

######################################
#     Tests for is_engine_config     #
######################################


def test_is_engine_config_true() -> None:
    assert is_engine_config({OBJECT_TARGET: "gravitorch.engines.AlphaEngine"})


def test_is_engine_config_false() -> None:
    assert not is_engine_config({OBJECT_TARGET: "torch.nn.Identity"})


##################################
#     Tests for setup_engine     #
##################################


def test_setup_engine_object() -> None:
    source = Mock(spec=BaseEngine)
    assert setup_engine(source) is source


def test_setup_engine_dict_mock() -> None:
    engine = Mock(spec=BaseEngine)
    factory_mock = Mock(return_value=engine)
    with patch("gravitorch.engines.base.BaseEngine.factory", factory_mock):
        assert setup_engine({OBJECT_TARGET: "name"}) == engine
        factory_mock.assert_called_once_with(_target_="name")


def test_setup_engine_dict() -> None:
    assert isinstance(
        setup_engine(
            {
                OBJECT_TARGET: "gravitorch.engines.AlphaEngine",
                "core_creator": {
                    OBJECT_TARGET: "gravitorch.creators.core.AdvancedCoreCreator",
                    "datasource_creator": {
                        OBJECT_TARGET: "gravitorch.creators.datasource.VanillaDataSourceCreator",
                        "config": {OBJECT_TARGET: "gravitorch.testing.DummyDataSource"},
                    },
                    "model_creator": {
                        OBJECT_TARGET: "gravitorch.creators.model.VanillaModelCreator",
                        "model_config": {
                            OBJECT_TARGET: "torch.nn.Linear",
                            "in_features": 8,
                            "out_features": 2,
                        },
                    },
                },
            }
        ),
        AlphaEngine,
    )


def test_setup_engine_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_engine({OBJECT_TARGET: "torch.nn.Identity"}), Identity)
        assert caplog.messages
