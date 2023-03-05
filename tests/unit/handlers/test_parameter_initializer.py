from unittest.mock import Mock

from pytest import mark

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import ParameterInitializer
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.parameter_initializers import BaseParameterInitializer

EVENTS = ("my_event", "my_other_event")


##########################################
#     Tests for ParameterInitializer     #
##########################################


def test_parameter_initializer_str() -> None:
    assert str(ParameterInitializer(parameter_initializer=Mock())).startswith(
        "ParameterInitializer("
    )


@mark.parametrize("event", EVENTS)
def test_parameter_initializer_event(event: str):
    assert ParameterInitializer(parameter_initializer=Mock(), event=event)._event == event


def test_parameter_initializer_event_default() -> None:
    assert ParameterInitializer(parameter_initializer=Mock())._event == EngineEvents.TRAIN_STARTED


@mark.parametrize("event", EVENTS)
def test_parameter_initializer_attach(event: str):
    parameter_initializer = Mock(spec=BaseParameterInitializer)
    handler = ParameterInitializer(parameter_initializer=parameter_initializer, event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(parameter_initializer.initialize, handler_kwargs={"engine": engine}),
    )


def test_parameter_initializer_attach_duplicate() -> None:
    handler = ParameterInitializer(parameter_initializer=Mock(spec=BaseParameterInitializer))
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()
