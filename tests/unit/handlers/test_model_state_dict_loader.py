from collections.abc import Sequence
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

import torch
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import ModelStateDictLoader, PartialModelStateDictLoader
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")
KEYS = (None, "my_key", ["key1", "key2"], ("key1", "key2"))


##########################################
#     Tests for ModelStateDictLoader     #
##########################################


def test_model_state_dict_loader_str(tmp_path: Path):
    assert str(ModelStateDictLoader(checkpoint_path=tmp_path)).startswith("ModelStateDictLoader(")


@mark.parametrize("event", EVENTS)
def test_model_state_dict_loader_event(tmp_path: Path, event: str):
    assert ModelStateDictLoader(checkpoint_path=tmp_path, event=event)._event == event


def test_model_state_dict_loader_event_default(tmp_path: Path):
    assert ModelStateDictLoader(checkpoint_path=tmp_path)._event == EngineEvents.STARTED


@mark.parametrize("strict", (True, False))
def test_model_state_dict_loader_strict(tmp_path: Path, strict: bool):
    assert ModelStateDictLoader(checkpoint_path=tmp_path, strict=strict)._strict == strict


def test_model_state_dict_loader_strict_default(tmp_path: Path):
    assert ModelStateDictLoader(checkpoint_path=tmp_path)._strict


@mark.parametrize("key", KEYS)
def test_model_state_dict_loader_key(
    tmp_path: Path, key: Union[str, list[str], tuple[str, ...], None]
):
    assert ModelStateDictLoader(checkpoint_path=tmp_path, key=key)._key == key


def test_model_state_dict_loader_key_default(tmp_path: Path):
    assert ModelStateDictLoader(checkpoint_path=tmp_path)._key is None


@mark.parametrize("event", EVENTS)
def test_model_state_dict_loader_attach(tmp_path: Path, event: str):
    handler = ModelStateDictLoader(checkpoint_path=tmp_path, event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(handler.load, handler_kwargs={"engine": engine}),
    )


def test_model_state_dict_loader_attach_duplicate(tmp_path: Path):
    handler = ModelStateDictLoader(checkpoint_path=tmp_path)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


@mark.parametrize("strict", (True, False))
@mark.parametrize("key", KEYS)
def test_model_state_dict_loader_load(
    tmp_path: Path, strict: bool, key: Union[str, list[str], tuple[str, ...], None]
):
    handler = ModelStateDictLoader(checkpoint_path=tmp_path, strict=strict, key=key)
    model = Mock(spec=nn.Module)
    with patch(
        "gravitorch.handlers.model_state_dict_loader.load_checkpoint_to_module"
    ) as load_mock:
        handler.load(Mock(spec=BaseEngine, model=model))
        load_mock.assert_called_once_with(
            path=tmp_path,
            module=model,
            strict=strict,
            key=key,
        )


#################################################
#     Tests for PartialModelStateDictLoader     #
#################################################


def test_partial_model_state_dict_loader_str(tmp_path: Path):
    assert str(PartialModelStateDictLoader(checkpoint_path=tmp_path)).startswith(
        "PartialModelStateDictLoader("
    )


@mark.parametrize("event", EVENTS)
def test_partial_model_state_dict_loader_event(tmp_path: Path, event: str):
    assert PartialModelStateDictLoader(checkpoint_path=tmp_path, event=event)._event == event


def test_partial_model_state_dict_loader_event_default(tmp_path: Path):
    assert PartialModelStateDictLoader(checkpoint_path=tmp_path)._event == EngineEvents.STARTED


@mark.parametrize("strict", (True, False))
def test_partial_model_state_dict_loader_strict(tmp_path: Path, strict: bool):
    assert PartialModelStateDictLoader(checkpoint_path=tmp_path, strict=strict)._strict == strict


def test_partial_model_state_dict_loader_strict_default(tmp_path: Path):
    assert PartialModelStateDictLoader(checkpoint_path=tmp_path)._strict


def test_partial_model_state_dict_loader_exclude_key_prefixes(tmp_path):
    assert (
        PartialModelStateDictLoader(
            checkpoint_path=tmp_path, exclude_key_prefixes="network.linear"
        )._exclude_key_prefixes
        == "network.linear"
    )


def test_partial_model_state_dict_loader_key_default(tmp_path: Path):
    assert PartialModelStateDictLoader(checkpoint_path=tmp_path)._exclude_key_prefixes == []


@mark.parametrize("event", EVENTS)
def test_partial_model_state_dict_loader_attach(tmp_path: Path, event: str):
    handler = PartialModelStateDictLoader(checkpoint_path=tmp_path, event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(handler.load, handler_kwargs={"engine": engine}),
    )


def test_partial_model_state_dict_loader_attach_duplicate(tmp_path: Path):
    handler = PartialModelStateDictLoader(checkpoint_path=tmp_path)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


@mark.parametrize("strict", (True, False))
@mark.parametrize("exclude_key_prefixes", ([], ("network.linear",)))
def test_partial_model_state_dict_loader_load_mock(
    tmp_path: Path, strict: bool, exclude_key_prefixes: Sequence
):
    handler = PartialModelStateDictLoader(
        checkpoint_path=tmp_path, strict=strict, exclude_key_prefixes=exclude_key_prefixes
    )
    model = Mock(spec=nn.Module)
    with patch("gravitorch.handlers.model_state_dict_loader.load_model_state_dict") as load_mock:
        handler.load(Mock(spec=BaseEngine, model=model))
        load_mock.assert_called_once_with(
            path=tmp_path,
            module=model,
            strict=strict,
            exclude_key_prefixes=exclude_key_prefixes,
        )


def test_partial_model_state_dict_loader_load(tmp_path: Path):
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(
        {
            "modules": {
                ct.MODEL: {
                    "weight": torch.ones(5, 4),
                    "bias": 2 * torch.ones(5),
                }
            }
        },
        checkpoint_path,
    )
    handler = PartialModelStateDictLoader(
        checkpoint_path=checkpoint_path, strict=False, exclude_key_prefixes=["bias"]
    )
    model = nn.Linear(4, 5)
    handler.load(Mock(spec=BaseEngine, model=model))
    assert not model.bias.equal(torch.ones(5))
    assert model.weight.equal(torch.ones(5, 4))
