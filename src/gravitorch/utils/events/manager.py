r"""This module implements an event manager."""

__all__ = ["EventManager"]

import logging
from collections import defaultdict
from typing import Optional

from gravitorch.utils.events.event_handlers import BaseEventHandler
from gravitorch.utils.format import str_indent, to_torch_sequence_str

logger = logging.getLogger(__name__)


class EventManager:
    r"""Implements an event manager.

    This event manager allows adding event handlers and firing events.
    An event is represented by a case-sensitive string.
    """

    def __init__(self):
        # This variable is used to store the handlers associated to each event.
        self._event_handlers = defaultdict(list)
        # This variable is used to track the last fired event name
        self._last_fired_event = None
        self.reset()

    def __repr__(self) -> str:
        if self._event_handlers:
            return (
                f"{self.__class__.__qualname__}(\n"
                f"  {str_indent(to_event_handlers_str(self._event_handlers))}\n"
                f"  last_fired_event={self._last_fired_event}\n"
                ")"
            )
        return (
            f"{self.__class__.__qualname__}(event_handlers=(), "
            f"last_fired_event={self._last_fired_event})"
        )

    @property
    def last_fired_event(self) -> Optional[str]:
        r"""Gets the last event name that was fired.

        Returns:
            str or ``None``: The last event name that was fired or
                ``None`` if no event was fired.
        """
        return self._last_fired_event

    def add_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        r"""Adds an event handler to an event.

        The event handler will be called everytime the event happens.

        Args:
            event (str): Specifies the event to attach the event
                handler.
            event_handler (``BaseEventHandler``): Specifies the
                event handler to attach to the event.

        Example usage:

        .. code-block:: python

            # Create an event manager
            >>> from gravitorch.utils.events import EventManager
            >>> event_manager = EventManager()
            # Add an event handler to the event manager
            >>> def hello_handler():
            ...     print('Hello!')
            ...
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> event_manager.add_event_handler('my_event', VanillaEventHandler(hello_handler))
        """
        self._event_handlers[str(event)].append(event_handler)
        logger.debug(f"Added {event_handler} to event {event}")

    def fire_event(self, event: str) -> None:
        r"""Fires the handler(s) for the given event.

        Args:
            event (str): Specifies the event to fire.

        Example usage:

        .. code-block:: python

            # Create an event manager
            >>> from gravitorch.utils.events import EventManager
            >>> event_manager = EventManager()
            # Fire the 'my_event' event
            >>> event_manager.fire_event('my_event')  # should do nothing because there is no event handler
            # Add an event handler
            >>> def hello_handler():
            ...     print("Hello!")
            ...
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> event_manager.add_event_handler('my_event', VanillaEventHandler(hello_handler))
            # Fire the 'my_event' event
            >>> event_manager.fire_event('my_event')
            Hello!
        """
        logger.debug(f"Firing {event} event")
        self._last_fired_event = event
        for event_handler in self._event_handlers[event]:
            event_handler.handle()

    def has_event_handler(
        self, event_handler: BaseEventHandler, event: Optional[str] = None
    ) -> bool:
        r"""Indicates if a handler is registered in the event manager.

        Note that this method relies on the ``__eq__`` method of the
        input event handler to compare event handlers.

        Args:
            event_handler (``BaseEventHandler``): Specifies the eventn
                handler to check.
            event (str or ``None``): Specifies an event to check. If
                the value is ``None``, it will check all the events.
                Default: ``None``

        Example usage:

        .. code-block:: python

            # Create an event manager
            >>> from gravitorch.utils.events import EventManager
            >>> event_manager = EventManager()
            # Define a handler
            >>> def hello_handler():
            ...     print("Hello!")
            ...
            # Check if `hello_handler` is registered in the event manager
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler))
            False
            # Check if `hello_handler` is registered in the event manager for 'my_event' event
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            False
            # Add an event handler
            >>> event_manager.add_event_handler('my_event', VanillaEventHandler(hello_handler))
            # Check if `hello_handler` is registered in the event manager
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler))
            True
            # Check if `hello_handler` is registered in the event manager for 'my_event' event
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            True
            # Check if `hello_handler` is registered in the event manager for 'my_other_event' event
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler), 'my_other_event')
            False
        """
        events = [event] if event else self._event_handlers
        for evnt in events:
            for evnt_handler in self._event_handlers[evnt]:
                if event_handler == evnt_handler:
                    return True
        return False

    def remove_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        r"""Removes an event handler of a given event.

        Note that if the same event handler was added multiple times
        the event, all the duplicated handlers are removed. This
        method relies on the ``__eq__`` method of the input event
        handler to compare event handlers.

        Args:
            event (str): Specifies the event handler is attached to.
            event_handler (``BaseEventHandler``): Specifies the event
                handler to remove.

        Raises:
            ValueError: if the event does not exist or if the handler
                is not attached to the event.

        Example usage:

        .. code-block:: python

            # Create an event manager
            >>> from gravitorch.utils.events import EventManager
            >>> event_manager = EventManager()
            # Add an event handler to the engine
            >>> def hello_handler():
            ...     print('Hello!')
            ...
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> event_manager.add_event_handler('my_event', VanillaEventHandler(hello_handler))
            # Check if `hello_handler` is registered in the event manager for 'my_event' event
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            True
            # Remove the event handler of the engine
            >>> event_manager.remove_event_handler('my_event', VanillaEventHandler(hello_handler))
            # Check if `hello_handler` is registered in the event manager for 'my_event' event
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            False
        """
        if event not in self._event_handlers:
            raise ValueError(f"'{event}' event does not exist")

        new_event_handlers = [
            handler for handler in self._event_handlers[event] if event_handler != handler
        ]
        if len(new_event_handlers) == len(self._event_handlers[event]):
            raise ValueError(
                f"{event_handler} is not found among registered event handlers for '{event}' event"
            )
        if len(new_event_handlers) > 0:
            self._event_handlers[event] = new_event_handlers
        else:
            del self._event_handlers[event]
        logger.debug(f"Removed {event_handler} in '{event}' event")

    def reset(self) -> None:
        r"""Resets the event manager.

        This method removes all the event handlers from the event manager.

        Example usage:

        .. code-block:: python

            # Create an event manager
            >>> from gravitorch.utils.events import EventManager
            >>> event_manager = EventManager()
            # Add an event handler to the engine
            >>> def hello_handler():
            ...     print('Hello!')
            ...
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> event_manager.add_event_handler('my_event', VanillaEventHandler(hello_handler))
            # Check if `hello_handler` is registered in the event manager for 'my_event' event
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            True
            >>> event_manager.fire_event('my_event')
            >>> event_manager.last_fired_event
            my_event
            # Reset the event manager
            >>> event_manager.reset()
            # Check if `hello_handler` is registered in the event manager for 'my_event' event
            >>> event_manager.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            False
            >>> event_manager.last_fired_event
            None
        """
        self._event_handlers.clear()
        self._last_fired_event = None


def to_event_handlers_str(event_handlers: dict[str, list], num_spaces: int = 2):
    r"""Computes a string representation of the event handlers.

    Args:
        event_handlers (dict): Specifies the dictionary with the list
            of handlers for each event.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
        str: A string representation of the event handlers
    """
    lines = []
    spaces = " " * num_spaces
    for key, value in event_handlers.items():
        lines.append(
            f"({key})\n{spaces}"
            f"{str_indent(to_torch_sequence_str(value, num_spaces), num_spaces)}"
        )
    return "\n".join(lines)
