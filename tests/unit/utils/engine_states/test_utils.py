from objectory import OBJECT_TARGET

from gravitorch.utils.engine_states import VanillaEngineState, setup_engine_state

########################################
#     Tests for setup_engine_state     #
########################################


def test_setup_engine_state_none():
    assert isinstance(setup_engine_state(None), VanillaEngineState)


def test_setup_engine_state_object():
    state = VanillaEngineState()
    assert setup_engine_state(state) is state


def test_setup_engine_state_dict():
    state = setup_engine_state(
        {OBJECT_TARGET: "gravitorch.utils.engine_states.VanillaEngineState", "max_epochs": 10}
    )
    assert isinstance(state, VanillaEngineState)
    assert state.max_epochs == 10
