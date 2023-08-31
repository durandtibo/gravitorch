from unittest.mock import Mock, patch

from minevent import EventHandler
from pytest import mark

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import ModelParameterAnalyzer

EVENTS = ("my_event", "my_other_event")


############################################
#     Tests for ModelParameterAnalyzer     #
############################################


def test_model_parameter_analyzer_str() -> None:
    assert str(ModelParameterAnalyzer()).startswith("ModelParameterAnalyzer(")


def test_model_parameter_analyzer_events() -> None:
    assert ModelParameterAnalyzer("my_event")._events == ("my_event",)


def test_model_parameter_analyzer_events_default() -> None:
    assert ModelParameterAnalyzer()._events == (EngineEvents.STARTED, EngineEvents.TRAIN_COMPLETED)


@mark.parametrize("tablefmt", ("rst", "github"))
def test_model_parameter_analyzer_tablefmt(tablefmt: str) -> None:
    assert ModelParameterAnalyzer(tablefmt=tablefmt)._tablefmt == tablefmt


def test_model_parameter_analyzer_tablefmt_default() -> None:
    assert ModelParameterAnalyzer()._tablefmt == "fancy_outline"


@mark.parametrize("floatfmt", (".6f", ".3f"))
def test_model_parameter_analyzer_floatfmt(floatfmt: str) -> None:
    assert ModelParameterAnalyzer(floatfmt=floatfmt)._floatfmt == floatfmt


def test_model_parameter_analyzer_floatfmt_default() -> None:
    assert ModelParameterAnalyzer()._floatfmt == ".6f"


@mark.parametrize("event", EVENTS)
def test_model_parameter_analyzer_attach(event: str) -> None:
    handler = ModelParameterAnalyzer(events=event)
    engine = Mock(spec=BaseEngine)
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        EventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


def test_model_parameter_analyzer_attach_2_events() -> None:
    handler = ModelParameterAnalyzer()
    engine = Mock(spec=BaseEngine)
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EngineEvents.STARTED,
        EventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EngineEvents.TRAIN_COMPLETED,
        EventHandler(handler.analyze, handler_kwargs={"engine": engine}),
    )


@mark.parametrize("tablefmt", ("rst", "github"))
@mark.parametrize("floatfmt", (".6f", ".3f"))
def test_model_parameter_analyzer_analyze(tablefmt: str, floatfmt: str) -> None:
    handler = ModelParameterAnalyzer(tablefmt=tablefmt, floatfmt=floatfmt)
    engine = Mock(spec=BaseEngine)
    with patch("gravitorch.handlers.model_parameter.show_parameter_summary") as analyze_mock:
        handler.analyze(engine)
        analyze_mock.assert_called_once_with(
            module=engine.model, tablefmt=tablefmt, floatfmt=floatfmt
        )
