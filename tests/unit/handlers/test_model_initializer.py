from unittest.mock import Mock

from pytest import mark

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import ModelInitializer
from gravitorch.nn.init import Constant
from gravitorch.utils.events import VanillaEventHandler

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
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(initializer.initialize, handler_kwargs={"engine": engine}),
    )


def test_model_initializer_attach_duplicate() -> None:
    handler = ModelInitializer(initializer=Constant(value=1.0))
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()
