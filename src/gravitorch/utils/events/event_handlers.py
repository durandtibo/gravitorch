__all__ = [
    "BaseEventHandler",
    "BaseEventHandlerWithArguments",
    "ConditionalEventHandler",
    "VanillaEventHandler",
]

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional


class BaseEventHandler(ABC):
    r"""Defines the base class to implement an event handler.

    A child class has to implement the following methods:

        - ``__eq__``
        - ``handle``
    """

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        r"""Compares two event handlers."""

    @abstractmethod
    def handle(self) -> None:
        r"""Handles the event."""


class BaseEventHandlerWithArguments(BaseEventHandler):
    r"""Defines a base class to implement an event handler with positional
    and/or keyword arguments.

    A child class has to implement the ``__eq__`` method.

    Args:
        handler (callable): Specifies the handler.
        handler_args (tuple or ``None``, optional): Specifies the
            positional argument of the handler. Default: ``None``
        handler_kwargs (dict or ``None``, optional): Specifies the
            arbitrary keyword arguments of the handler.
            Default: ``None``
    """

    def __init__(
        self,
        handler: Callable,
        handler_args: Optional[tuple] = None,
        handler_kwargs: Optional[dict] = None,
    ):
        self._handler = handler
        self._handler_args = handler_args or tuple()
        self._handler_kwargs = handler_kwargs or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(handler={self._handler}, "
            f"handler_args={self._handler_args}, handler_kwargs={self._handler_kwargs})"
        )

    @property
    def handler(self) -> Callable:
        r"""callable: The handler."""
        return self._handler

    @property
    def handler_args(self) -> tuple:
        r"""tuple: Variable length argument list of the handler."""
        return self._handler_args

    @property
    def handler_kwargs(self) -> dict:
        r"""dict: Arbitrary keyword arguments of the handler."""
        return self._handler_kwargs

    def handle(self) -> None:
        r"""Handles the event."""
        self._handler(*self._handler_args, **self._handler_kwargs)


class VanillaEventHandler(BaseEventHandlerWithArguments):
    r"""Implements a simple event handler."""

    def __eq__(self, other: Any) -> bool:
        r"""Compares two event handlers."""
        if isinstance(other, VanillaEventHandler):
            return (
                self.handler == other.handler
                and self.handler_args == other.handler_args
                and self.handler_kwargs == other.handler_kwargs
            )
        return False


class ConditionalEventHandler(BaseEventHandlerWithArguments):
    r"""Implements a conditional event handler.

    The handler is executed only if the condition is ``True``.

    Args:
        handler (callable): Specifies the handler.
        condition (callable): Specifies the condition for this event
            handler. The condition should be callable without
            arguments.
        handler_args (tuple or ``None``): Specifies the positional
            argument of the handler.
        handler_kwargs (dict): Specifies the arbitrary keyword
            arguments of the handler.
    """

    def __init__(
        self,
        handler: Callable,
        condition: Callable,
        handler_args: Optional[tuple] = None,
        handler_kwargs: Optional[dict] = None,
    ):
        super().__init__(handler=handler, handler_args=handler_args, handler_kwargs=handler_kwargs)
        if not callable(condition):
            raise ValueError(f"The condition is not callable (received: {condition})")
        self._condition = condition

    def __eq__(self, other: Any) -> bool:
        r"""Compares two event handlers."""
        if isinstance(other, ConditionalEventHandler):
            return (
                self.handler == other.handler
                and self.handler_args == other.handler_args
                and self.handler_kwargs == other.handler_kwargs
                and self.condition == other.condition
            )
        return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(handler={self._handler}, "
            f"handler_args={self._handler_args}, handler_kwargs={self._handler_kwargs}, "
            f"condition={self._condition})"
        )

    @property
    def condition(self) -> Callable:
        r"""callable: The condition."""
        return self._condition

    def handle(self) -> None:
        r"""Handles the event."""
        if self._condition():
            super().handle()
