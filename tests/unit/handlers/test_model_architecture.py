from unittest.mock import Mock, patch

from pytest import mark

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.handlers import ModelArchitectureAnalyzer
from gravitorch.handlers.model_architecture import NetworkArchitectureAnalyzer

EVENTS = ("my_event", "my_other_event")


###############################################
#     Tests for ModelArchitectureAnalyzer     #
###############################################


def test_model_architecture_analyzer_str() -> None:
    assert str(ModelArchitectureAnalyzer()).startswith("ModelArchitectureAnalyzer(")


def test_model_architecture_analyzer_events() -> None:
    assert ModelArchitectureAnalyzer("my_event")._events == ("my_event",)


def test_model_architecture_analyzer_events_default() -> None:
    assert ModelArchitectureAnalyzer()._events == (EngineEvents.STARTED,)


@mark.parametrize("event", EVENTS)
def test_model_architecture_analyzer_attach(event: str) -> None:
    handler = ModelArchitectureAnalyzer(events=event)
    engine = Mock(spec=BaseEngine)
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


def test_model_architecture_analyzer_attach_2_events() -> None:
    handler = ModelArchitectureAnalyzer(EVENTS)
    engine = Mock(spec=BaseEngine)
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EVENTS[0],
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EVENTS[1],
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


def test_model_architecture_analyzer_analyze() -> None:
    handler = ModelArchitectureAnalyzer()
    engine = Mock(spec=BaseEngine)
    with patch("gravitorch.handlers.model_architecture.analyze_model_architecture") as analyze_mock:
        handler.analyze(engine)
        analyze_mock.assert_called_once_with(model=engine.model, engine=engine)


#################################################
#     Tests for NetworkArchitectureAnalyzer     #
#################################################


def test_network_architecture_analyzer_analyze() -> None:
    handler = NetworkArchitectureAnalyzer()
    engine = Mock(spec=BaseEngine)
    with patch(
        "gravitorch.handlers.model_architecture.analyze_network_architecture"
    ) as analyze_mock:
        handler.analyze(engine)
        analyze_mock.assert_called_once_with(model=engine.model, engine=engine)
