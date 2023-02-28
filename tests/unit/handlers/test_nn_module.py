from unittest.mock import Mock

from pytest import mark
from torch.nn import Linear, ModuleDict, Sequential

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import ModelModuleFreezer
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")


########################################
#     Tests for ModelModuleFreezer     #
########################################


def test_model_module_freezer_str():
    assert str(ModelModuleFreezer(module_name="module")).startswith("ModelModuleFreezer(")


@mark.parametrize("event", EVENTS)
def test_model_module_freezer_event(event: str):
    assert ModelModuleFreezer(module_name="module", event=event)._event == event


def test_model_module_freezer_event_default():
    assert ModelModuleFreezer(module_name="module")._event == EngineEvents.TRAIN_STARTED


@mark.parametrize("event", EVENTS)
def test_model_module_freezer_attach(event: str):
    handler = ModelModuleFreezer(module_name="module", event=event)
    engine = Mock(spec=BaseEngine)
    engine.has_event_handler.return_value = False
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(
            handler.freeze,
            handler_kwargs={"engine": engine},
        ),
    )


def test_model_module_freezer_attach_duplicate():
    handler = ModelModuleFreezer(module_name="module")
    engine = Mock(spec=BaseEngine)
    engine.has_event_handler.return_value = True
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_model_module_freezer_freeze():
    handler = ModelModuleFreezer(module_name="network.0")
    engine = Mock(
        spec=BaseEngine, model=ModuleDict({"network": Sequential(Linear(4, 4), Linear(4, 4))})
    )
    handler.freeze(engine)
    print(engine.model)
    for param in engine.model.network[0].parameters():
        param.requires_grad = True
    for param in engine.model.network[1].parameters():
        param.requires_grad = False
