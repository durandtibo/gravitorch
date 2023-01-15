__all__ = ["add_unique_event_handler", "setup_handler", "setup_and_attach_handlers", "to_events"]

import logging
from typing import Union

from gravitorch.engines.base import BaseEngine
from gravitorch.handlers.base import BaseHandler
from gravitorch.utils.events import BaseEventHandler
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def add_unique_event_handler(
    engine: BaseEngine, event: str, event_handler: BaseEventHandler
) -> None:
    r"""Adds an event handler to the engine if it was not added previously.

    This function checks if the event handler was already added to the
    engine. If not, the event handler is added to the engine otherwise
    it is not.

    Args:
        engine (``BaseEngine``): Specifies the engine used to attach
            the event handler.
        event (str): Specifies the event.
        event_handler (``BaseEventHandler``): Specifies the event
            handler.

    Example usage:

    .. code-block:: python

        # Create an engine
        >>> from gravitorch.engines import AlphaEngine
        >>> engine = AlphaEngine(...)
        # Create an event handler
        >>> def hello_handler():
        ...     print('Hello!')
        ...
        >>> from gravitorch.utils.events import VanillaEventHandler
        >>> event_handler = VanillaEventHandler(hello_handler)
        # Add an event handler to the engine
        >>> from gravitorch.handlers import add_unique_event_handler
        >>> add_unique_event_handler(engine, 'my_event', event_handler)
    """
    if engine.has_event_handler(event_handler, event):
        logger.info(f"{event_handler} is already added to the engine for '{event}' event")
    else:
        logger.info(f"Adding {event_handler} to '{event}' event")
        engine.add_event_handler(event, event_handler)


def setup_handler(handler: Union[BaseHandler, dict]) -> BaseHandler:
    r"""Sets up a handler.

    Args:
        handler (``BaseHandler`` or dict): Specifies the handler or
            its configuration.

    Returns:
        ``BaseHandler``: The handler.
    """
    if isinstance(handler, dict):
        logger.info(
            f"Initializing a handler from its configuration... {str_target_object(handler)}"
        )
        handler = BaseHandler.factory(**handler)
    return handler


def setup_and_attach_handlers(
    engine: BaseEngine,
    handlers: Union[tuple[Union[BaseHandler, dict], ...], list[Union[BaseHandler, dict]]],
) -> list[BaseHandler]:
    r"""Sets up and attaches some handlers to the engine.

    Note that if you call this function ``N`` times with the same
    handlers, the handlers will be attached ``N`` times to the engine.

    Args:
        engine (``BaseEngine``): Specifies the engine.
        handlers (list or tuple): Specifies the list of handlers or
            their configuration.

    Returns:
        list: The list of handlers attached to the engine.
    """
    list_handlers = []
    for handler in handlers:
        handler = setup_handler(handler)
        list_handlers.append(handler)
        handler.attach(engine)
    return list_handlers


def to_events(events: Union[str, tuple[str, ...], list[str]]) -> tuple[str, ...]:
    r"""Converts the input events to a tuple of events.

    If the input is a string (i.e. single event), it is converted to a
    tuple with a single event.

    Args:
        events (str or tuple or list): Specifies the input events.

    Returns:
        tuple: The tuple of events.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.handlers import to_events
        >>> to_events('my_event')
        ('my_event',)
        >>> to_events(('my_event', 'my_other_event'))
        ('my_event', 'my_other_event')
        >>> to_events(['my_event', 'my_other_event'])
        ('my_event', 'my_other_event')
    """
    if isinstance(events, str):
        return (events,)
    return tuple(events)
