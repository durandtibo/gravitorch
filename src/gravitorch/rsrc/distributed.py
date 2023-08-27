from __future__ import annotations

__all__ = ["DistributedContext"]

import logging
from types import TracebackType

from gravitorch.distributed.comm import BACKEND_TO_CONTEXT, resolve_backend
from gravitorch.distributed.utils import show_distributed_context_info
from gravitorch.rsrc.base import BaseResource

logger = logging.getLogger(__name__)


class DistributedContext(BaseResource):
    r"""Implements a context manager to initialize the distributed
    backend.

    Args:
    ----
        backend (str or ``None``, optional): Specifies the distributed
            backend. If ``'auto'``, this function will find the best
            option for the distributed backend according to the
            context and some rules. Default: ``'auto'``
        log_info (bool, optional): If ``True``, information about the
            resource is logged after the resource is initialized.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.distributed import backend
        >>> from gravitorch.rsrc import DistributedContext
        >>> with DistributedContext(backend=None):
        ...     backend()
        ...
    """

    def __init__(self, backend: str | None = "auto", log_info: bool = False) -> None:
        self._backend = resolve_backend(backend)
        self._context = BACKEND_TO_CONTEXT[self._backend]

        self._log_info = bool(log_info)

    def __enter__(self) -> DistributedContext:
        logger.info(f"Initialing `{self._backend}` distributed context...")
        self._context.__enter__()
        if self._log_info:
            show_distributed_context_info()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.info(f"Exiting `{self._backend}` distributed context...")
        self._context.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(backend={self._backend}, log_info={self._log_info})"
