r"""This module implements some utility functions for the engine
states."""

from __future__ import annotations

__all__ = ["setup_engine_state"]

import logging

from gravitorch.utils.engine_states.base import BaseEngineState
from gravitorch.utils.engine_states.vanilla import EngineState
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_engine_state(state: BaseEngineState | dict | None) -> BaseEngineState:
    r"""Sets up the engine state.

    The state is instantiated from its configuration by using the
    ``BaseEngineState`` factory function.

    Args:
    ----
        state (``BaseEngineState`` or dict or None): Specifies the
            engine state or its configuration. If ``None``, the
            ``EngineState`` is instantiated.

    Returns:
    -------
        ``BaseEngineState``: The engine state.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.engine_states import setup_engine_state
        >>> state = setup_engine_state({"_target_": "gravitorch.utils.engine_states.EngineState"})
        >>> state
        EngineState(
          (modules): AssetManager(num_assets=0)
          (histories): HistoryManager()
          (random_seed): 9984043075503325450
          (max_epochs): 1
          (epoch): -1
          (iteration): -1
        )
    """
    if state is None:
        state = EngineState()
    if isinstance(state, dict):
        logger.info(
            f"Initializing an engine state from its configuration... {str_target_object(state)}"
        )
        state = BaseEngineState.factory(**state)
    return state
