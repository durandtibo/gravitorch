from unittest.mock import Mock

import torch
from pytest import mark
from torch.nn import Linear, Module

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.handlers import ModelInitializer
from gravitorch.nn.init import Constant, constant
from gravitorch.testing import create_dummy_engine

EVENTS = ("my_event", "my_other_event")


######################################
#     Tests for ModelInitializer     #
######################################


def test_model_initializer_str() -> None:
    assert str(ModelInitializer(initializer=Mock())).startswith("ModelInitializer(")


@mark.parametrize("event", EVENTS)
def test_model_initializer_event(event: str) -> None:
    assert ModelInitializer(initializer=Mock(), event=event)._event == event


def test_model_initializer_event_default() -> None:
    assert ModelInitializer(initializer=Mock())._event == EngineEvents.TRAIN_STARTED


@mark.parametrize("event", EVENTS)
def test_model_initializer_attach(event: str) -> None:
    initializer = Constant(value=1.0)
    handler = ModelInitializer(initializer=initializer, event=event)
    model = Mock(spec=Module)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False), model=model)
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(initializer.initialize, handler_kwargs={"module": model}),
    )


def test_model_initializer_attach_duplicate() -> None:
    handler = ModelInitializer(initializer=Constant(value=1.0))
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_model_initializer_initialize() -> None:
    engine = create_dummy_engine(model=Linear(4, 6))
    constant(engine.model, value=0.0)
    ModelInitializer(initializer=Constant(value=1.0)).attach(engine)
    engine.fire_event(EngineEvents.TRAIN_STARTED)
    assert engine.model.weight.equal(torch.ones(6, 4))
    assert engine.model.bias.equal(torch.ones(6))
