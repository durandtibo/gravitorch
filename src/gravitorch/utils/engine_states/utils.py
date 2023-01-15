r"""This module implements some utility functions for the engine states."""

__all__ = ["setup_engine_state"]

import logging
from typing import Union

from gravitorch.utils.engine_states.base import BaseEngineState
from gravitorch.utils.engine_states.vanilla import VanillaEngineState
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_engine_state(state: Union[BaseEngineState, dict, None]) -> BaseEngineState:
    r"""Sets up the engine state.

    The state is instantiated from its configuration by using the
    ``BaseEngineState`` factory function.

    Args:
        state (``BaseEngineState`` or dict or None): Specifies the
            engine state or its configuration. If ``None``, the
            ``VanillaEngineState`` is instantiated.

    Returns:
        ``BaseEngineState``: The engine state.
    """
    if state is None:
        state = VanillaEngineState()
    if isinstance(state, dict):
        logger.info(
            f"Initializing an engine state from its configuration... {str_target_object(state)}"
        )
        state = BaseEngineState.factory(**state)
    return state
