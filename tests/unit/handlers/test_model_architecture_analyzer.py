from unittest.mock import Mock, patch

from pytest import mark

from gravitorch.engines import EngineEvents
from gravitorch.handlers import ModelArchitectureAnalyzer
from gravitorch.handlers.model_architecture_analyzer import (
    ModelNetworkArchitectureAnalyzer,
)
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")


###############################################
#     Tests for ModelArchitectureAnalyzer     #
###############################################


def test_model_architecture_analyzer_str():
    assert str(ModelArchitectureAnalyzer()).startswith("ModelArchitectureAnalyzer(")


def test_model_architecture_analyzer_events():
    assert ModelArchitectureAnalyzer("my_event")._events == ("my_event",)


def test_model_architecture_analyzer_events_default():
    assert ModelArchitectureAnalyzer()._events == (EngineEvents.STARTED,)


@mark.parametrize("event", EVENTS)
def test_model_architecture_analyzer_attach(event: str):
    handler = ModelArchitectureAnalyzer(events=event)
    engine = Mock()
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


def test_model_architecture_analyzer_attach_2_events():
    handler = ModelArchitectureAnalyzer(EVENTS)
    engine = Mock()
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EVENTS[0],
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EVENTS[1],
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


def test_model_architecture_analyzer_analyze():
    handler = ModelArchitectureAnalyzer()
    engine = Mock()
    with patch(
        "gravitorch.handlers.model_architecture_analyzer.analyze_model_architecture"
    ) as analyze_mock:
        handler.analyze(engine)
        analyze_mock.assert_called_once_with(model=engine.model, engine=engine)


######################################################
#     Tests for ModelNetworkArchitectureAnalyzer     #
######################################################


def test_model_network_architecture_analyzer_analyze():
    handler = ModelNetworkArchitectureAnalyzer()
    engine = Mock()
    with patch(
        "gravitorch.handlers.model_architecture_analyzer.analyze_model_network_architecture"
    ) as analyze_mock:
        handler.analyze(engine)
        analyze_mock.assert_called_once_with(model=engine.model, engine=engine)
