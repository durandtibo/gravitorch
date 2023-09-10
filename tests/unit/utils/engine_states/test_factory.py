from objectory import OBJECT_TARGET

from gravitorch.utils.engine_states import EngineState, setup_engine_state

########################################
#     Tests for setup_engine_state     #
########################################


def test_setup_engine_state_none() -> None:
    assert isinstance(setup_engine_state(None), EngineState)


def test_setup_engine_state_object() -> None:
    state = EngineState()
    assert setup_engine_state(state) is state


def test_setup_engine_state_dict() -> None:
    state = setup_engine_state(
        {OBJECT_TARGET: "gravitorch.utils.engine_states.EngineState", "max_epochs": 10}
    )
    assert isinstance(state, EngineState)
    assert state.max_epochs == 10
