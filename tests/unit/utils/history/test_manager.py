import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, mark

from gravitorch.utils.history import (
    GenericHistory,
    HistoryManager,
    MaxScalarHistory,
    MinScalarHistory,
)

NAMES = ("NAME", "my_history")


####################################
#     Tests for HistoryManager     #
####################################


def test_history_manager_str():
    assert str(HistoryManager()) == "HistoryManager()"


def test_history_manager_str_with_history():
    manager = HistoryManager()
    manager.get_history("something")
    assert str(manager).startswith("HistoryManager(")


def test_history_manager_len_empty():
    assert len(HistoryManager()) == 0


def test_history_manager_len_1_history():
    manager = HistoryManager()
    manager.get_history("my_history1")
    assert len(manager) == 1


def test_history_manager_len_2_histories():
    manager = HistoryManager()
    manager.get_history("my_history1")
    manager.get_history("my_history2")
    assert len(manager) == 2


@mark.parametrize("key", NAMES)
def test_history_manager_add_history_with_key(key: str):
    manager = HistoryManager()
    manager.add_history(MinScalarHistory("loss"), key)
    assert key in manager._histories


@mark.parametrize("key", NAMES)
def test_history_manager_add_history_without_key(key: str):
    manager = HistoryManager()
    manager.add_history(MinScalarHistory(key))
    assert key in manager._histories


def test_history_manager_add_history_duplicate_key(caplog: LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        manager = HistoryManager()
        manager.add_history(MinScalarHistory("loss"))
        manager.add_history(MinScalarHistory("loss"))
        assert "loss" in manager._histories
        assert len(caplog.messages) == 1


def test_history_manager_get_best_values_empty():
    assert HistoryManager().get_best_values() == {}


def test_history_manager_get_best_values_empty_history():
    state = HistoryManager()
    history = MinScalarHistory("loss")
    state.add_history(history)
    assert state.get_best_values() == {}


def test_history_manager_get_best_values_not_comparable_history():
    state = HistoryManager()
    history = GenericHistory("loss")
    history.add_value(1.2, step=0)
    history.add_value(0.8, step=1)
    state.add_history(history)
    assert state.get_best_values() == {}


def test_history_manager_get_best_values_1_history():
    state = HistoryManager()
    history = MinScalarHistory("loss")
    state.add_history(history)
    history.add_value(1.2, step=0)
    history.add_value(0.8, step=1)
    assert state.get_best_values() == {"loss": 0.8}


def test_history_manager_get_best_values_2_history():
    state = HistoryManager()
    history1 = MinScalarHistory("loss")
    history1.add_value(1.2, step=0)
    history1.add_value(0.8, step=1)
    state.add_history(history1)
    history2 = MaxScalarHistory("accuracy")
    history2.add_value(42, step=0)
    history2.add_value(41, step=1)
    state.add_history(history2)
    assert state.get_best_values() == {"loss": 0.8, "accuracy": 42}


def test_history_manager_get_history_exists():
    manager = HistoryManager()
    history = MinScalarHistory("loss")
    manager.add_history(history)
    assert manager.get_history("loss") is history


def test_history_manager_get_history_does_not_exist():
    manager = HistoryManager()
    history = manager.get_history("loss")
    assert isinstance(history, GenericHistory)
    assert len(history.get_recent_history()) == 0


def test_history_manager_get_histories():
    manager = HistoryManager()
    history1 = MinScalarHistory("loss")
    history2 = MinScalarHistory("accuracy")
    manager.add_history(history1)
    manager.add_history(history2)
    assert manager.get_histories() == {"loss": history1, "accuracy": history2}


def test_history_manager_get_histories_empty():
    assert HistoryManager().get_histories() == {}


def test_history_manager_has_history_true():
    manager = HistoryManager()
    manager.add_history(MinScalarHistory("loss"))
    assert manager.has_history("loss")


def test_history_manager_has_history_false():
    assert not HistoryManager().has_history("loss")


def test_history_manager_load_state_dict_empty():
    manager = HistoryManager()
    manager.load_state_dict({})
    assert len(manager) == 0


def test_history_manager_load_state_dict_with_existing_history():
    manager = HistoryManager()
    manager.add_history(MinScalarHistory("loss"))
    manager.get_history("loss").add_value(2, step=0)
    manager.load_state_dict(
        {
            "loss": {
                "state": {
                    "history": ((0, 10), (1, 9), (2, 8), (3, 7), (4, 6)),
                    "improved": True,
                    "best_value": 6,
                },
            }
        }
    )
    history = manager.get_history("loss")
    assert isinstance(history, MinScalarHistory)
    assert history.get_last_value() == 6
    assert history.get_best_value() == 6


def test_history_manager_load_state_dict_without_history():
    manager = HistoryManager()
    manager.load_state_dict(
        {
            "loss": {
                "config": {
                    OBJECT_TARGET: "gravitorch.utils.history.comparable.MinScalarHistory",
                    "name": "loss",
                    "max_size": 10,
                },
                "state": {
                    "history": ((0, 10), (1, 9), (2, 8), (3, 7), (4, 6)),
                    "improved": True,
                    "best_value": 6,
                },
            }
        },
    )
    history = manager.get_history("loss")
    assert isinstance(history, MinScalarHistory)
    assert history.get_last_value() == 6
    assert history.get_best_value() == 6


def test_history_manager_state_dict_empty():
    assert HistoryManager().state_dict() == {}


def test_history_manager_state_dict_1_history():
    manager = HistoryManager()
    history = MinScalarHistory("loss")
    manager.add_history(history)
    assert manager.state_dict() == {"loss": history.to_dict()}


def test_history_manager_state_dict_2_history():
    manager = HistoryManager()
    history1 = MinScalarHistory("loss")
    manager.add_history(history1)
    history2 = MaxScalarHistory("accuracy")
    manager.add_history(history2)
    assert manager.state_dict() == {"loss": history1.to_dict(), "accuracy": history2.to_dict()}
