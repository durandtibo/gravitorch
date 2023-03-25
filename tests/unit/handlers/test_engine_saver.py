import logging
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.handlers import (
    BestEngineStateSaver,
    BestHistorySaver,
    EpochEngineStateSaver,
    LastHistorySaver,
    TagEngineStateSaver,
)
from gravitorch.utils.history import GenericHistory, MaxScalarHistory

EVENTS = ("my_event", "my_other_event")


######################################
#     Tests for BestHistorySaver     #
######################################


def test_best_history_saver_str(tmp_path: Path) -> None:
    assert str(BestHistorySaver(tmp_path)).startswith("BestHistorySaver(")


def test_best_history_saver_path(tmp_path: Path) -> None:
    assert BestHistorySaver(tmp_path)._path == tmp_path


@mark.parametrize("event", EVENTS)
def test_best_history_saver_event(tmp_path: Path, event: str) -> None:
    assert BestHistorySaver(tmp_path, event=event)._event == event


def test_best_history_saver_event_default(tmp_path: Path) -> None:
    assert BestHistorySaver(tmp_path)._event == EngineEvents.COMPLETED


@mark.parametrize("event", EVENTS)
def test_best_history_saver_attach(tmp_path: Path, event: str) -> None:
    saver = BestHistorySaver(tmp_path, event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    saver.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(saver.save, handler_kwargs={"engine": engine}),
    )


def test_best_history_saver_attach_duplicate(tmp_path: Path) -> None:
    handler = BestHistorySaver(tmp_path, event="my_event")
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_best_history_saver_save_only_main_process_true_main_process(tmp_path: Path) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, get_histories=Mock(return_value={}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: True):
        with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
            saver = BestHistorySaver(tmp_path)
            saver.save(engine)
            save_mock.assert_called_once_with({}, tmp_path.joinpath("history_best.pt"))


def test_best_history_saver_save_only_main_process_true_not_main_process(tmp_path: Path) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, get_histories=Mock(return_value={}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: False):
        with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
            saver = BestHistorySaver(tmp_path)
            saver.save(engine)
            save_mock.assert_not_called()


@mark.parametrize("main_process", (True, False))
def test_best_history_saver_save_only_main_process_false(
    tmp_path: Path, main_process: bool
) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, get_histories=Mock(return_value={}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: main_process):
        with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
            saver = BestHistorySaver(tmp_path, only_main_process=False)
            saver.save(engine)
            save_mock.assert_called_once_with({}, tmp_path.joinpath("history_best.pt"))


######################################
#     Tests for LastHistorySaver     #
######################################


def test_last_history_saver_str(tmp_path: Path) -> None:
    assert str(LastHistorySaver(tmp_path)).startswith("LastHistorySaver(")


def test_last_history_saver_path(tmp_path: Path) -> None:
    assert LastHistorySaver(tmp_path)._path == tmp_path


@mark.parametrize("event", EVENTS)
def test_last_history_saver_event(tmp_path: Path, event: str) -> None:
    assert LastHistorySaver(tmp_path, event=event)._event == event


def test_last_history_saver_event_default(tmp_path: Path) -> None:
    assert LastHistorySaver(tmp_path)._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("event", EVENTS)
def test_last_history_saver_attach(tmp_path: Path, event: str) -> None:
    saver = LastHistorySaver(tmp_path, event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    saver.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(saver.save, handler_kwargs={"engine": engine}),
    )


def test_last_history_saver_attach_duplicate(tmp_path: Path) -> None:
    handler = LastHistorySaver(tmp_path, event="my_event")
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_last_history_saver_save_only_main_process_true_main_process(tmp_path: Path) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, get_histories=Mock(return_value={}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: True):
        with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
            saver = LastHistorySaver(tmp_path)
            saver.save(engine)
            save_mock.assert_called_once_with({}, tmp_path.joinpath("history_last.pt"))


def test_last_history_saver_save_only_main_process_true_not_main_process(tmp_path: Path) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, get_histories=Mock(return_value={}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: False):
        with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
            saver = LastHistorySaver(tmp_path)
            saver.save(engine)
            save_mock.assert_not_called()


@mark.parametrize("main_process", (True, False))
def test_last_history_saver_save_only_main_process_false(
    tmp_path: Path, main_process: bool
) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, get_histories=Mock(return_value={}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: main_process):
        with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
            saver = LastHistorySaver(tmp_path, only_main_process=False)
            saver.save(engine)
            save_mock.assert_called_once_with({}, tmp_path.joinpath("history_last.pt"))


##########################################
#     Tests for BestEngineStateSaver     #
##########################################


def test_best_engine_state_saver_str(tmp_path: Path) -> None:
    assert str(BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))).startswith(
        "BestEngineStateSaver("
    )


def test_best_engine_state_saver_path(tmp_path: Path) -> None:
    assert BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))._path == tmp_path


@mark.parametrize("event", EVENTS)
def test_best_engine_state_saver_event(tmp_path: Path, event: str) -> None:
    assert BestEngineStateSaver(tmp_path, event=event, keys=("loss", "accuracy"))._event == event


def test_best_engine_state_saver_event_default(tmp_path: Path) -> None:
    assert (
        BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))._event
        == EngineEvents.EPOCH_COMPLETED
    )


@mark.parametrize("keys", (("loss", "accuracy"), ["loss", "accuracy"]))
def test_best_engine_state_saver_keys(
    tmp_path: Path, keys: Union[tuple[str, ...], list[str]]
) -> None:
    assert BestEngineStateSaver(tmp_path, keys=keys)._keys == ("loss", "accuracy")


@mark.parametrize("event", EVENTS)
def test_best_engine_state_saver_attach(tmp_path: Path, event: str) -> None:
    saver = BestEngineStateSaver(tmp_path, event=event, keys=("loss", "accuracy"))
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    saver.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(saver.save, handler_kwargs={"engine": engine}),
    )


def test_best_engine_state_saver_attach_duplicate(tmp_path: Path) -> None:
    handler = BestEngineStateSaver(tmp_path, event="my_event", keys=("loss", "accuracy"))
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_best_engine_state_saver_save_only_main_process_true_main_process(tmp_path: Path) -> None:
    save_mock = Mock()
    engine = Mock(
        spec=BaseEngine,
        state_dict=Mock(return_value={"key": 123}),
        has_history=Mock(return_value=True),
        get_history=lambda key: MaxScalarHistory(
            key, elements=[(0, 1)], improved=True, best_value=1
        ),
    )
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: True):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 1):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))
                saver.save(engine)
                assert save_mock.call_args_list == [
                    (({"key": 123}, tmp_path.joinpath("best_loss/ckpt_engine_1.pt")), {}),
                    (({"key": 123}, tmp_path.joinpath("best_accuracy/ckpt_engine_1.pt")), {}),
                ]


def test_best_engine_state_saver_save_only_main_process_true_not_main_process(
    tmp_path: Path,
) -> None:
    save_mock = Mock()
    engine = Mock(
        spec=BaseEngine,
        state_dict=Mock(return_value={"key": 123}),
        has_history=Mock(return_value=True),
        get_history=lambda key: MaxScalarHistory(
            key, elements=[(0, 1)], improved=True, best_value=1
        ),
    )
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: False):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 0):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))
                saver.save(engine)
                save_mock.assert_not_called()


@mark.parametrize("main_process", (True, False))
def test_best_engine_state_saver_save_only_main_process_false(
    tmp_path: Path, main_process: bool
) -> None:
    save_mock = Mock()
    engine = Mock(
        spec=BaseEngine,
        state_dict=Mock(return_value={"key": 123}),
        has_history=Mock(return_value=True),
        get_history=lambda key: MaxScalarHistory(
            key, elements=[(0, 1)], improved=True, best_value=1
        ),
    )
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: main_process):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 1):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = BestEngineStateSaver(
                    tmp_path, only_main_process=False, keys=("loss", "accuracy")
                )
                saver.save(engine)
                assert save_mock.call_args_list == [
                    (({"key": 123}, tmp_path.joinpath("best_loss/ckpt_engine_1.pt")), {}),
                    (({"key": 123}, tmp_path.joinpath("best_accuracy/ckpt_engine_1.pt")), {}),
                ]


def test_best_engine_state_saver_save_no_history(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    saver = BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))
    engine = Mock(spec=BaseEngine, has_history=Mock(return_value=False))
    with patch("gravitorch.handlers.engine_saver.save_pytorch") as save_mock:
        with caplog.at_level(logging.WARNING):
            saver.save(engine)
            assert len(caplog.messages) == 2
            save_mock.assert_not_called()


def test_best_engine_state_saver_save_non_comparable_history(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    saver = BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))
    engine = Mock(
        spec=BaseEngine,
        has_history=Mock(return_value=True),
        get_history=lambda key: GenericHistory(key),
    )
    with patch("gravitorch.handlers.engine_saver.save_pytorch") as save_mock:
        with caplog.at_level(logging.WARNING):
            saver.save(engine)
            assert len(caplog.messages) == 2
            save_mock.assert_not_called()


def test_best_engine_state_saver_save_history_has_not_improved(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    saver = BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))
    engine = Mock(
        spec=BaseEngine,
        has_history=Mock(return_value=True),
        get_history=lambda key: MaxScalarHistory(
            key, elements=[(0, 2), (1, 1)], improved=False, best_value=2
        ),
    )
    with patch("gravitorch.handlers.engine_saver.save_pytorch") as save_mock:
        with caplog.at_level(logging.WARNING):
            saver.save(engine)
            assert len(caplog.messages) == 0
            save_mock.assert_not_called()


def test_best_engine_state_saver_save_empty_history(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    saver = BestEngineStateSaver(tmp_path, keys=("loss", "accuracy"))
    engine = Mock(
        spec=BaseEngine,
        has_history=Mock(return_value=True),
        get_history=lambda key: MaxScalarHistory(key),
    )
    with patch("gravitorch.handlers.engine_saver.save_pytorch") as save_mock:
        with caplog.at_level(logging.WARNING):
            saver.save(engine)
            assert len(caplog.messages) == 2
            save_mock.assert_not_called()


###########################################
#     Tests for EpochEngineStateSaver     #
###########################################


def test_epoch_engine_state_saver_str(tmp_path: Path) -> None:
    assert str(EpochEngineStateSaver(tmp_path)).startswith("EpochEngineStateSaver(")


def test_epoch_engine_state_saver_path(tmp_path: Path) -> None:
    assert EpochEngineStateSaver(tmp_path)._path == tmp_path


@mark.parametrize("event", EVENTS)
def test_epoch_engine_state_saver_event(tmp_path: Path, event: str) -> None:
    assert EpochEngineStateSaver(tmp_path, event=event)._event == event


def test_epoch_engine_state_saver_event_default(tmp_path: Path) -> None:
    assert EpochEngineStateSaver(tmp_path)._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("event", EVENTS)
def test_epoch_engine_state_saver_attach(tmp_path: Path, event: str) -> None:
    saver = EpochEngineStateSaver(tmp_path, event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    saver.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(saver.save, handler_kwargs={"engine": engine}),
    )


def test_epoch_engine_state_saver_attach_duplicate(tmp_path: Path) -> None:
    handler = EpochEngineStateSaver(tmp_path, event="my_event")
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_epoch_engine_state_saver_save_only_main_process_true_main_process(tmp_path: Path) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, epoch=0, state_dict=Mock(return_value={"key": 123}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: True):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 1):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = EpochEngineStateSaver(tmp_path)
                saver.save(engine)
                save_mock.assert_called_once_with(
                    {"key": 123}, tmp_path.joinpath("epoch/ckpt_engine_0_1.pt")
                )


def test_epoch_engine_state_saver_save_only_main_process_true_not_main_process(
    tmp_path: Path,
) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, epoch=0, state_dict=Mock(return_value={"key": 123}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: False):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 1):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = EpochEngineStateSaver(tmp_path)
                saver.save(engine)
                save_mock.assert_not_called()


@mark.parametrize("main_process", (True, False))
def test_epoch_engine_state_saver_save_only_main_process_false(
    tmp_path: Path, main_process: bool
) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, epoch=0, state_dict=Mock(return_value={"key": 123}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: main_process):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 1):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = EpochEngineStateSaver(tmp_path, only_main_process=False)
                saver.save(engine)
                save_mock.assert_called_once_with(
                    {"key": 123}, tmp_path.joinpath("epoch/ckpt_engine_0_1.pt")
                )


#########################################
#     Tests for TagEngineStateSaver     #
#########################################


def test_tag_engine_state_saver_str(tmp_path: Path) -> None:
    assert str(TagEngineStateSaver(tmp_path)).startswith("TagEngineStateSaver(")


def test_tag_engine_state_saver_path(tmp_path: Path) -> None:
    assert TagEngineStateSaver(tmp_path)._path == tmp_path


@mark.parametrize("event", EVENTS)
def test_tag_engine_state_saver_event(tmp_path: Path, event: str) -> None:
    assert TagEngineStateSaver(tmp_path, event=event)._event == event


def test_tag_engine_state_saver_event_default(tmp_path: Path) -> None:
    assert TagEngineStateSaver(tmp_path)._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("tag", ("tag1", "tag2"))
def test_tag_engine_state_saver_tag(tmp_path: Path, tag: str) -> None:
    assert TagEngineStateSaver(tmp_path, tag=tag)._tag == tag


def test_tag_engine_state_saver_tag_default(tmp_path: Path) -> None:
    assert TagEngineStateSaver(tmp_path)._tag == "last"


@mark.parametrize("event", EVENTS)
def test_tag_engine_state_saver_attach(tmp_path: Path, event: str) -> None:
    saver = TagEngineStateSaver(tmp_path, event=event)
    engine = Mock(has_event_handler=Mock(return_value=False))
    saver.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(saver.save, handler_kwargs={"engine": engine}),
    )


def test_tag_engine_state_saver_attach_duplicate(tmp_path: Path) -> None:
    handler = TagEngineStateSaver(tmp_path, event="my_event")
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_tag_engine_state_saver_save_only_main_process_true_main_process(tmp_path: Path) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, state_dict=Mock(return_value={"key": 123}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: True):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 1):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = TagEngineStateSaver(tmp_path)
                saver.save(engine)
                save_mock.assert_called_once_with(
                    {"key": 123}, tmp_path.joinpath("last/ckpt_engine_1.pt")
                )


def test_tag_engine_state_saver_save_only_main_process_true_not_main_process(
    tmp_path: Path,
) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, state_dict=Mock(return_value={"key": 123}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: False):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 0):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = TagEngineStateSaver(tmp_path)
                saver.save(engine)
                save_mock.assert_not_called()


@mark.parametrize("main_process", (True, False))
def test_tag_engine_state_saver_save_only_main_process_false(
    tmp_path: Path, main_process: bool
) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, state_dict=Mock(return_value={"key": 123}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: main_process):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 1):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = TagEngineStateSaver(tmp_path, only_main_process=False)
                saver.save(engine)
                save_mock.assert_called_once_with(
                    {"key": 123}, tmp_path.joinpath("last/ckpt_engine_1.pt")
                )


@mark.parametrize("tag", ("tag1", "tag2"))
def test_tag_engine_state_saver_save_custom_tag(tmp_path: Path, tag: str) -> None:
    save_mock = Mock()
    engine = Mock(spec=BaseEngine, state_dict=Mock(return_value={"key": 123}))
    with patch("gravitorch.handlers.engine_saver.dist.is_main_process", lambda *args: True):
        with patch("gravitorch.handlers.engine_saver.dist.get_rank", lambda *args: 0):
            with patch("gravitorch.handlers.engine_saver.save_pytorch", save_mock):
                saver = TagEngineStateSaver(tmp_path, tag=tag)
                saver.save(engine)
                save_mock.assert_called_once_with(
                    {"key": 123}, tmp_path.joinpath(f"{tag}/ckpt_engine_0.pt")
                )
