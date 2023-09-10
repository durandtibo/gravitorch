from pytest import mark, raises
from torch import nn

from gravitorch.utils.engine_states import EngineState
from gravitorch.utils.history import GenericHistory, MinScalarHistory

NAMES = ("name1", "name2")


#################################
#     Tests for EngineState     #
#################################


def test_engine_state_str() -> None:
    assert str(EngineState()).startswith("EngineState(")


def test_engine_state_default_values() -> None:
    state = EngineState()
    assert state.epoch == -1
    assert state.iteration == -1
    assert state.max_epochs == 1
    assert state.random_seed == 9984043075503325450
    assert len(state._histories) == 0
    assert len(state._modules) == 0


@mark.parametrize("epoch", (1, 2))
def test_engine_state_epoch(epoch: int) -> None:
    assert EngineState(epoch=epoch).epoch == epoch


@mark.parametrize("iteration", (1, 2))
def test_engine_state_iteration(iteration: int) -> None:
    assert EngineState(iteration=iteration).iteration == iteration


@mark.parametrize("max_epochs", (1, 2))
def test_engine_state_max_epochs(max_epochs: int) -> None:
    assert EngineState(max_epochs=max_epochs).max_epochs == max_epochs


@mark.parametrize("random_seed", (1, 2))
def test_engine_state_random_seed(random_seed: int) -> None:
    assert EngineState(random_seed=random_seed).random_seed == random_seed


@mark.parametrize("key", NAMES)
def test_engine_state_add_history_with_key(key: str) -> None:
    state = EngineState()
    state.add_history(MinScalarHistory("loss"), key)
    assert state._histories.has_history(key)


@mark.parametrize("key", NAMES)
def test_engine_state_add_history_without_key(key: str) -> None:
    state = EngineState()
    state.add_history(MinScalarHistory(key))
    assert state._histories.has_history(key)


@mark.parametrize("name", NAMES)
def test_engine_state_add_module(name: str) -> None:
    state = EngineState()
    state.add_module(name, nn.Linear(4, 5))
    assert state._modules.has_module(name)


def test_engine_state_get_best_values_without_history() -> None:
    assert EngineState().get_best_values() == {}


def test_engine_state_get_best_values_with_history() -> None:
    state = EngineState()
    history = MinScalarHistory("loss")
    state.add_history(history)
    history.add_value(1.2, step=0)
    history.add_value(0.8, step=1)
    assert state.get_best_values() == {"loss": 0.8}


def test_engine_state_get_history_exists() -> None:
    state = EngineState()
    history = MinScalarHistory("loss")
    state.add_history(history)
    assert state.get_history("loss") is history


def test_engine_state_get_history_does_not_exist() -> None:
    state = EngineState()
    history = state.get_history("loss")
    assert isinstance(history, GenericHistory)
    assert len(history.get_recent_history()) == 0


def test_engine_state_get_histories() -> None:
    manager = EngineState()
    history1 = MinScalarHistory("loss")
    history2 = MinScalarHistory("accuracy")
    manager.add_history(history1)
    manager.add_history(history2)
    assert manager.get_histories() == {"loss": history1, "accuracy": history2}


def test_engine_state_get_histories_empty() -> None:
    assert EngineState().get_histories() == {}


def test_engine_state_get_module() -> None:
    state = EngineState()
    state.add_module("my_module", nn.Linear(4, 5))
    assert isinstance(state.get_module("my_module"), nn.Linear)


def test_engine_state_get_module_missing() -> None:
    state = EngineState()
    with raises(ValueError, match="The module 'my_module' does not exist"):
        state.get_module("my_module")


def test_engine_state_has_history_true() -> None:
    state = EngineState()
    state.add_history(MinScalarHistory("loss"))
    assert state.has_history("loss")


def test_engine_state_has_history_false() -> None:
    assert not EngineState().has_history("loss")


def test_engine_state_has_module_true() -> None:
    state = EngineState()
    state.add_module("my_module", nn.Linear(4, 5))
    assert state.has_module("my_module")


def test_engine_state_has_module_false() -> None:
    assert not EngineState().has_module("my_module")


def test_engine_state_increment_epoch_1() -> None:
    state = EngineState()
    assert state.epoch == -1
    state.increment_epoch()
    assert state.epoch == 0


def test_engine_state_increment_epoch_2() -> None:
    state = EngineState()
    assert state.epoch == -1
    state.increment_epoch(2)
    assert state.epoch == 1


def test_engine_state_increment_iteration_1() -> None:
    state = EngineState()
    assert state.iteration == -1
    state.increment_iteration()
    assert state.iteration == 0


def test_engine_state_increment_iteration_2() -> None:
    state = EngineState()
    assert state.iteration == -1
    state.increment_iteration(2)
    assert state.iteration == 1


def test_engine_state_load_state_dict() -> None:
    state = EngineState()
    state.load_state_dict(
        {
            "epoch": 5,
            "iteration": 101,
            "histories": {},
            "modules": {},
        }
    )
    assert state.epoch == 5
    assert state.iteration == 101


def test_engine_state_state_dict_load_state_dict() -> None:
    state = EngineState()
    state.increment_epoch(6)
    state_dict = state.state_dict()
    state.increment_epoch(2)
    state.load_state_dict(state_dict)
    assert state.epoch == 5


def test_engine_state_remove_module_exists() -> None:
    state = EngineState()
    state.add_module("my_module", nn.Linear(4, 5))
    assert state.has_module("my_module")
    state.remove_module("my_module")
    assert not state.has_module("my_module")


def test_engine_state_remove_module_does_not_exist() -> None:
    state = EngineState()
    with raises(
        ValueError, match="The module 'my_module' does not exist so it is not possible to remove it"
    ):
        state.remove_module("my_module")


def test_engine_state_state_dict() -> None:
    assert EngineState().state_dict() == {
        "epoch": -1,
        "iteration": -1,
        "histories": {},
        "modules": {},
    }
