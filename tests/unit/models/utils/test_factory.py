from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from torch import nn

from gravitorch.engines import BaseEngine
from gravitorch.models import is_model_config
from gravitorch.models.utils import (
    attach_module_to_engine,
    setup_and_attach_model,
    setup_model,
)

#############################################
#     Tests for attach_module_to_engine     #
#############################################


def test_attach_module_to_engine_attach() -> None:
    engine = Mock(spec=BaseEngine)
    module = Mock()
    attach_module_to_engine(module, engine)
    module.attach.assert_called_once_with(engine)


def test_attach_module_to_engine_no_attach() -> None:
    engine = Mock(spec=BaseEngine)
    module = Mock()
    with patch("gravitorch.models.utils.factory.hasattr", lambda *args, **kwargs: False):
        attach_module_to_engine(module, engine)
        module.attach.assert_not_called()


#####################################
#     Tests for is_model_config     #
#####################################


def test_is_model_config_true() -> None:
    assert is_model_config({"_target_": "gravitorch.models.VanillaModel"})


def test_is_model_config_false() -> None:
    assert not is_model_config({"_target_": "torch.nn.Identity"})


#################################
#     Tests for setup_model     #
#################################


def test_setup_model_object() -> None:
    model = nn.Linear(4, 6)
    assert setup_model(model) is model


def test_setup_model_dict(tmp_path: Path) -> None:
    assert isinstance(
        setup_model({OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}),
        nn.Linear,
    )


############################################
#     Tests for setup_and_attach_model     #
############################################


def test_setup_and_attach_model_with_attach() -> None:
    engine = Mock(spec=BaseEngine)
    model = Mock()
    assert setup_and_attach_model(engine, model) is model
    model.attach.assert_called_once_with(engine=engine)


def test_setup_and_attach_model_without_attach() -> None:
    engine = Mock(spec=BaseEngine)
    model = nn.Linear(4, 6)
    assert setup_and_attach_model(engine, model) is model
