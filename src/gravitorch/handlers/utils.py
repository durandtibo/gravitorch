from __future__ import annotations

__all__ = [
    "add_unique_event_handler",
    "is_handler_config",
    "setup_and_attach_handlers",
    "setup_handler",
    "to_events",
]

import logging
from typing import TYPE_CHECKING

from minevent import BaseEventHandler
from objectory.utils import is_object_config

from gravitorch.handlers.base import BaseHandler
from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


def add_unique_event_handler(
    engine: BaseEngine, event: str, event_handler: BaseEventHandler
) -> None:
    r"""Adds an event handler to the engine if it was not added
    previously.

    This function checks if the event handler was already added to the
    engine. If not, the event handler is added to the engine otherwise
    it is not.

    Args:
    ----
        engine (``BaseEngine``): Specifies the engine used to attach
            the event handler.
        event (str): Specifies the event.
        event_handler (``BaseEventHandler``): Specifies the event
            handler.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.events import GEventHandler
        >>> from gravitorch.handlers import add_unique_event_handler
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> def hello_handler():
        ...     print("Hello!")
        ...
        >>> event_handler = GEventHandler(hello_handler)
        >>> add_unique_event_handler(engine, "my_event", event_handler)
    """
    if engine.has_event_handler(event_handler, event):
        logger.info(f"{event_handler} is already added to the engine for '{event}' event")
    else:
        logger.info(f"Adding {event_handler} to '{event}' event")
        engine.add_event_handler(event, event_handler)


def is_handler_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseHandler``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseHandler`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import is_handler_config
        >>> is_handler_config({"_target_": "gravitorch.handlers.EpochLRMonitor"})
        True
    """
    return is_object_config(config, BaseHandler)


def setup_handler(handler: BaseHandler | dict) -> BaseHandler:
    r"""Sets up a handler.

    Args:
    ----
        handler (``BaseHandler`` or dict): Specifies the handler or
            its configuration.

    Returns:
    -------
        ``BaseHandler``: The handler.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import setup_handler
        >>> handler = setup_handler({"_target_": "gravitorch.handlers.EpochLRMonitor"})
        >>> handler
        EpochLRMonitor(freq=1, event=train_epoch_started)
    """
    if isinstance(handler, dict):
        logger.info(
            f"Initializing a handler from its configuration... {str_target_object(handler)}"
        )
        handler = BaseHandler.factory(**handler)
    if not isinstance(handler, BaseHandler):
        logger.warning(f"handler is not a BaseHandler (received: {type(handler)})")
    return handler


def setup_and_attach_handlers(
    engine: BaseEngine,
    handlers: tuple[BaseHandler | dict, ...] | list[BaseHandler | dict],
) -> list[BaseHandler]:
    r"""Sets up and attaches some handlers to the engine.

    Note that if you call this function ``N`` times with the same
    handlers, the handlers will be attached ``N`` times to the engine.

    Args:
    ----
        engine (``BaseEngine``): Specifies the engine.
        handlers (list or tuple): Specifies the list of handlers or
            their configuration.

    Returns:
    -------
        list: The list of handlers attached to the engine.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import setup_and_attach_handlers
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handlers = setup_and_attach_handlers(
        ...     engine, [{"_target_": "gravitorch.handlers.EpochLRMonitor"}]
        ... )
        >>> handlers
        [EpochLRMonitor(freq=1, event=train_epoch_started)]
    """
    list_handlers = []
    for handler in handlers:
        handler = setup_handler(handler)
        list_handlers.append(handler)
        handler.attach(engine)
    return list_handlers


def to_events(events: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    r"""Converts the input events to a tuple of events.

    If the input is a string (i.e. single event), it is converted to a
    tuple with a single event.

    Args:
    ----
        events (str or tuple or list): Specifies the input events.

    Returns:
    -------
        tuple: The tuple of events.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import to_events
        >>> to_events("my_event")
        ('my_event',)
        >>> to_events(("my_event", "my_other_event"))
        ('my_event', 'my_other_event')
        >>> to_events(["my_event", "my_other_event"])
        ('my_event', 'my_other_event')
    """
    if isinstance(events, str):
        return (events,)
    return tuple(events)
