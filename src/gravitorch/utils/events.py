from __future__ import annotations

__all__ = ["GEventHandler", "GConditionalEventHandler"]

from coola.utils import str_indent, str_mapping
from minevent import ConditionalEventHandler, EventHandler


class GEventHandler(EventHandler):
    r"""Implements a variant of ``minvent.EventHandler`` to not show the
    arguments in the to string method.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.events import GEventHandler
        >>> def hello_handler() -> None:
        ...     print("Hello!")
        ...
        >>> handler = GEventHandler(hello_handler)
        >>> print(repr(handler))  # doctest:+ELLIPSIS
        GEventHandler(
          (handler): <function hello_handler at 0x...>
          (handler_args): ()
          (handler_kwargs): {}
        )
        >>> print(str(handler))  # doctest:+ELLIPSIS
        GEventHandler(
          (handler): <function hello_handler at 0x...>
        )
        >>> handler.handle()
        Hello!
    """

    def __str__(self) -> str:
        args = str_indent(str_mapping({"handler": self._handler}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class GConditionalEventHandler(ConditionalEventHandler):
    r"""Implements a variant of ``minvent.GConditionalEventHandler`` to
    not show the arguments in the to string method.

    Example usage:

    .. code-block:: pycon

        >>> from minevent import PeriodicCondition
        >>> from gravitorch.utils.events import GConditionalEventHandler
        >>> def hello_handler() -> None:
        ...     print("Hello!")
        ...
        >>> handler = GConditionalEventHandler(hello_handler, PeriodicCondition(freq=3))
        >>> print(repr(handler))  # doctest:+ELLIPSIS
        GConditionalEventHandler(
          (handler): <function hello_handler at 0x...>
          (handler_args): ()
          (handler_kwargs): {}
          (condition): PeriodicCondition(freq=3, step=0)
        )
        >>> print(str(handler))  # doctest:+ELLIPSIS
        GConditionalEventHandler(
          (handler): <function hello_handler at 0x...>
          (condition): PeriodicCondition(freq=3, step=0)
        )
        >>> handler.handle()
        Hello!
        >>> handler.handle()
        >>> handler.handle()
        >>> handler.handle()
        Hello!
    """

    def __str__(self) -> str:
        args = str_indent(str_mapping({"handler": self._handler, "condition": self._condition}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
