from pathlib import Path
from typing import Union
from unittest.mock import Mock

import torch
from minevent import EventHandler
from pytest import mark, raises

from gravitorch.engines import BaseEngine
from gravitorch.handlers import (
    EngineStateLoader,
    EngineStateLoaderWithExcludeKeys,
    EngineStateLoaderWithIncludeKeys,
)
from gravitorch.utils.io import save_pytorch

EVENTS = ("my_event", "my_other_event")


#######################################
#     Tests for EngineStateLoader     #
#######################################


def test_engine_state_loader_str(tmp_path: Path) -> None:
    assert str(EngineStateLoader(tmp_path, event="my_event")).startswith("EngineStateLoader(")


def test_engine_state_loader_path(tmp_path: Path) -> None:
    assert EngineStateLoader(tmp_path, event="my_event")._path == tmp_path


@mark.parametrize("event", EVENTS)
def test_engine_state_loader_event(tmp_path: Path, event: str) -> None:
    assert EngineStateLoader(tmp_path, event=event)._event == event


@mark.parametrize("missing_ok", (True, False))
def test_engine_state_loader_missing_ok(tmp_path: Path, missing_ok: bool) -> None:
    assert (
        EngineStateLoader(tmp_path, event="my_event", missing_ok=missing_ok)._missing_ok
        == missing_ok
    )


@mark.parametrize("event", EVENTS)
def test_engine_state_loader_attach(tmp_path: Path, event: str) -> None:
    loader = EngineStateLoader(tmp_path, event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    engine.has_event_handler.return_value = False
    loader.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        EventHandler(loader.load_engine_state_dict, handler_kwargs={"engine": engine}),
    )


def test_engine_state_loader_attach_duplicate(tmp_path: Path) -> None:
    handler = EngineStateLoader(tmp_path, event="my_event")
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_engine_state_loader_load_engine_state_dict_file_exists(tmp_path: Path) -> None:
    path = tmp_path.joinpath("state.pt")
    save_pytorch({"key", 1}, path)
    loader = EngineStateLoader(path, event="my_event")
    engine = Mock(spec=BaseEngine)
    loader.load_engine_state_dict(engine)
    engine.load_state_dict.assert_called_once_with({"key", 1})


def test_engine_state_loader_load_engine_state_dict_file_exists_tensor(tmp_path: Path) -> None:
    path = tmp_path.joinpath("state.pt")
    save_pytorch({"key", torch.ones(2, 3)}, path)
    loader = EngineStateLoader(path, event="my_event")
    engine = Mock(spec=BaseEngine)
    loader.load_engine_state_dict(engine)
    engine.load_state_dict.assert_called_once()


def test_engine_state_loader_load_engine_state_dict_file_does_not_exist_missing_ok_true(
    tmp_path: Path,
) -> None:
    loader = EngineStateLoader(tmp_path, event="my_event", missing_ok=True)
    engine = Mock(spec=BaseEngine)
    loader.load_engine_state_dict(engine)
    engine.load_state_dict.assert_not_called()


def test_engine_state_loader_load_engine_state_dict_file_does_not_exist_missing_ok_false(
    tmp_path: Path,
) -> None:
    loader = EngineStateLoader(tmp_path, event="my_event")
    with raises(FileNotFoundError):
        loader.load_engine_state_dict(engine=Mock(spec=BaseEngine))


######################################################
#     Tests for EngineStateLoaderWithExcludeKeys     #
######################################################


def test_engine_state_loader_with_exclude_keys_str(tmp_path: Path) -> None:
    assert str(
        EngineStateLoaderWithExcludeKeys(tmp_path, event="my_event", exclude_keys=("key1", "key2"))
    ).startswith("EngineStateLoaderWithExcludeKeys(")


@mark.parametrize("exclude_keys", (("key1", "key2"), ["key1", "key2"]))
def test_engine_state_loader_with_exclude_keys_include_keys(
    tmp_path: Path,
    exclude_keys: Union[tuple[str, ...], list[str]],
) -> None:
    assert EngineStateLoaderWithExcludeKeys(
        tmp_path, event="my_event", exclude_keys=exclude_keys
    )._exclude_keys == ("key1", "key2")


def test_engine_state_loader_with_exclude_keys_prepare_state_dict_remove_extra_keys(
    tmp_path: Path,
) -> None:
    loader = EngineStateLoaderWithExcludeKeys(
        tmp_path, event="my_event", exclude_keys=("key3", "key4")
    )
    assert loader._prepare_state_dict({"key1": 1, "key2": 2, "key3": 3}) == {"key1": 1, "key2": 2}


def test_engine_state_loader_with_exclude_keys_prepare_state_dict_all_keys_removed(
    tmp_path: Path,
) -> None:
    loader = EngineStateLoaderWithExcludeKeys(
        tmp_path, event="my_event", exclude_keys=("key1", "key2")
    )
    assert loader._prepare_state_dict({"key1": 1, "key2": 2}) == {}


def test_engine_state_loader_with_exclude_keys_load_engine_state_dict_file_exists(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("state.pt")
    save_pytorch({"key1": 1, "key2": 2, "key3": 3}, path)
    loader = EngineStateLoaderWithExcludeKeys(path, event="my_event", exclude_keys=("key3", "key4"))
    engine = Mock(spec=BaseEngine)
    loader.load_engine_state_dict(engine)
    engine.load_state_dict.assert_called_once_with({"key1": 1, "key2": 2})


######################################################
#     Tests for EngineStateLoaderWithIncludeKeys     #
######################################################


def test_engine_state_loader_with_include_keys_str(tmp_path: Path) -> None:
    assert str(
        EngineStateLoaderWithIncludeKeys(tmp_path, event="my_event", include_keys=("key1", "key2"))
    ).startswith("EngineStateLoaderWithIncludeKeys(")


@mark.parametrize("include_keys", (("key1", "key2"), ["key1", "key2"]))
def test_engine_state_loader_with_include_keys_include_keys(
    tmp_path: Path,
    include_keys: Union[tuple[str, ...], list[str]],
) -> None:
    assert EngineStateLoaderWithIncludeKeys(
        tmp_path, event="my_event", include_keys=include_keys
    )._include_keys == ("key1", "key2")


def test_engine_state_loader_with_include_keys_prepare_state_dict_remove_extra_keys(
    tmp_path: Path,
) -> None:
    loader = EngineStateLoaderWithIncludeKeys(
        tmp_path, event="my_event", include_keys=("key1", "key2")
    )
    assert loader._prepare_state_dict({"key1": 1, "key2": 2, "key3": 3}) == {"key1": 1, "key2": 2}


def test_engine_state_loader_with_include_keys_prepare_state_dict_partial_keys(
    tmp_path: Path,
) -> None:
    loader = EngineStateLoaderWithIncludeKeys(
        tmp_path, event="my_event", include_keys=("key1", "key2")
    )
    assert loader._prepare_state_dict({"key1": 1, "key3": 3}) == {"key1": 1}


def test_engine_state_loader_with_include_keys_prepare_state_dict_all_keys_missing(
    tmp_path: Path,
) -> None:
    loader = EngineStateLoaderWithIncludeKeys(
        tmp_path, event="my_event", include_keys=("key1", "key2")
    )
    assert loader._prepare_state_dict({"key3": 1, "key4": 2, "key5": 3}) == {}


def test_engine_state_loader_with_include_keys_load_engine_state_dict_file_exists(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("state.pt")
    save_pytorch({"key1": 1, "key2": 2, "key3": 3}, path)
    loader = EngineStateLoaderWithIncludeKeys(path, event="my_event", include_keys=("key1", "key2"))
    engine = Mock(spec=BaseEngine)
    loader.load_engine_state_dict(engine)
    engine.load_state_dict.assert_called_once_with({"key1": 1, "key2": 2})
