from unittest.mock import Mock

from pytest import mark

from gravitorch.engines import EngineEvents
from gravitorch.handlers import ParameterInitializerHandler
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")


#################################################
#     Tests for ParameterInitializerHandler     #
#################################################


def test_parameter_initializer_handler_str():
    assert str(ParameterInitializerHandler(parameter_initializer=Mock())).startswith(
        "ParameterInitializerHandler("
    )


@mark.parametrize("event", EVENTS)
def test_parameter_initializer_handler_event(event: str):
    assert ParameterInitializerHandler(parameter_initializer=Mock(), event=event)._event == event


def test_parameter_initializer_handler_event_default():
    assert (
        ParameterInitializerHandler(parameter_initializer=Mock())._event
        == EngineEvents.TRAIN_STARTED
    )


@mark.parametrize("event", EVENTS)
def test_parameter_initializer_handler_attach(event: str):
    parameter_initializer = Mock()
    handler = ParameterInitializerHandler(parameter_initializer=parameter_initializer, event=event)
    engine = Mock()
    engine.has_event_handler.return_value = False
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(parameter_initializer.initialize, handler_kwargs={"engine": engine}),
    )


def test_parameter_initializer_handler_attach_duplicate():
    handler = ParameterInitializerHandler(parameter_initializer=Mock())
    engine = Mock()
    engine.has_event_handler.return_value = True
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()
