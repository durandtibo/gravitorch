from unittest.mock import Mock

from minevent import EventHandler
from pytest import mark
from torch.nn import Linear, ModuleDict, Sequential

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import ModelFreezer

EVENTS = ("my_event", "my_other_event")


##################################
#     Tests for ModelFreezer     #
##################################


def test_model_freezer_str() -> None:
    assert str(ModelFreezer()).startswith("ModelFreezer(")


@mark.parametrize("event", EVENTS)
def test_model_freezer_event(event: str) -> None:
    assert ModelFreezer(event=event)._event == event


def test_model_freezer_event_default() -> None:
    assert ModelFreezer()._event == EngineEvents.TRAIN_STARTED


@mark.parametrize("event", EVENTS)
def test_model_freezer_attach(event: str) -> None:
    handler = ModelFreezer(event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        EventHandler(
            handler.freeze,
            handler_kwargs={"engine": engine},
        ),
    )


def test_model_freezer_attach_duplicate() -> None:
    handler = ModelFreezer()
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_model_freezer_freeze() -> None:
    handler = ModelFreezer()
    engine = Mock(
        spec=BaseEngine, model=ModuleDict({"network": Sequential(Linear(4, 4), Linear(4, 4))})
    )
    handler.freeze(engine)
    for param in engine.model.parameters():
        param.requires_grad = False


def test_model_freezer_freeze_submodule() -> None:
    handler = ModelFreezer(module_name="network.0")
    engine = Mock(
        spec=BaseEngine, model=ModuleDict({"network": Sequential(Linear(4, 4), Linear(4, 4))})
    )
    handler.freeze(engine)
    for param in engine.model.network[0].parameters():
        param.requires_grad = True
    for param in engine.model.network[1].parameters():
        param.requires_grad = False
