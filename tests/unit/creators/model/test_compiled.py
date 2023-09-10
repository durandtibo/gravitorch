from typing import Union
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch import nn

from gravitorch.creators.model import (
    BaseModelCreator,
    CompiledModelCreator,
    ModelCreator,
)
from gravitorch.engines import BaseEngine

##########################################
#     Tests for CompiledModelCreator     #
##########################################


@fixture
def model_creator() -> BaseModelCreator:
    return ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}
    )


def test_compiled_model_creator_str() -> None:
    assert str(CompiledModelCreator(model_creator=Mock())).startswith("CompiledModelCreator(")


@mark.parametrize(
    "model_creator",
    (
        ModelCreator(
            model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}
        ),
        {
            OBJECT_TARGET: "gravitorch.creators.model.ModelCreator",
            "model_config": {OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6},
        },
    ),
)
def test_compiled_model_creator_model_creator(model_creator: Union[BaseModelCreator, dict]) -> None:
    assert isinstance(
        CompiledModelCreator(model_creator=model_creator)._model_creator,
        ModelCreator,
    )


def test_compiled_model_creator_compile_kwargs_default(model_creator: BaseModelCreator) -> None:
    assert CompiledModelCreator(model_creator=model_creator)._compile_kwargs == {}


def test_compiled_model_creator_compile_kwargs(
    model_creator: BaseModelCreator,
) -> None:
    assert CompiledModelCreator(
        model_creator=model_creator, compile_kwargs={"mode": "default"}
    )._compile_kwargs == {"mode": "default"}


def test_compiled_model_creator_create() -> None:
    model = nn.Linear(4, 6)
    creator = CompiledModelCreator(
        model_creator=Mock(spec=BaseModelCreator, create=Mock(return_value=model)),
    )
    with patch("gravitorch.creators.model.compiled.torch.compile") as compile_mock:
        creator.create(engine=Mock(spec=BaseEngine))
        compile_mock.assert_called_once_with(model)


def test_compiled_model_creator_create_compile_kwargs() -> None:
    model = nn.Linear(4, 6)
    creator = CompiledModelCreator(
        model_creator=Mock(spec=BaseModelCreator, create=Mock(return_value=model)),
        compile_kwargs={"mode": "default"},
    )
    with patch("gravitorch.creators.model.compiled.torch.compile") as compile_mock:
        creator.create(engine=Mock(spec=BaseEngine))
        compile_mock.assert_called_once_with(model, mode="default")
