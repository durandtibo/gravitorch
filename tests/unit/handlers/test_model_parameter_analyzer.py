from unittest.mock import Mock, patch

from pytest import mark

from gravitorch.engines import EngineEvents
from gravitorch.handlers import ModelParameterAnalyzer
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")


############################################
#     Tests for ModelParameterAnalyzer     #
############################################


def test_model_parameter_analyzer_str():
    assert str(ModelParameterAnalyzer()).startswith("ModelParameterAnalyzer(")


def test_model_parameter_analyzer_events():
    assert ModelParameterAnalyzer("my_event")._events == ("my_event",)


def test_model_parameter_analyzer_events_default():
    assert ModelParameterAnalyzer()._events == (EngineEvents.STARTED, EngineEvents.TRAIN_COMPLETED)


@mark.parametrize("tablefmt", ("rst", "github"))
def test_model_parameter_analyzer_tablefmt(tablefmt: str):
    assert ModelParameterAnalyzer(tablefmt=tablefmt)._tablefmt == tablefmt


def test_model_parameter_analyzer_tablefmt_default():
    assert ModelParameterAnalyzer()._tablefmt == "rst"


@mark.parametrize("event", EVENTS)
def test_model_parameter_analyzer_attach(event: str):
    handler = ModelParameterAnalyzer(events=event)
    engine = Mock()
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


def test_model_parameter_analyzer_attach_2_events():
    handler = ModelParameterAnalyzer()
    engine = Mock()
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EngineEvents.STARTED,
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EngineEvents.TRAIN_COMPLETED,
        VanillaEventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


@mark.parametrize("tablefmt", ("rst", "github"))
def test_model_parameter_analyzer_analyze(tablefmt: str):
    handler = ModelParameterAnalyzer(tablefmt=tablefmt)
    engine = Mock()
    with patch("gravitorch.handlers.model_parameter_analyzer.show_parameter_stats") as analyze_mock:
        handler.analyze(engine)
        analyze_mock.assert_called_once_with(module=engine.model, tablefmt=tablefmt)
