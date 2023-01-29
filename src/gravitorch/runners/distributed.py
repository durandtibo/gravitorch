r"""This module defines a base class to implement a distributed runner."""

__all__ = ["BaseDistributedRunner", "BaseEngineDistributedRunner"]

import logging
from abc import abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Optional, Union

from gravitorch.distributed import comm as dist
from gravitorch.distributed.auto import auto_dist_backend, auto_distributed_context
from gravitorch.distributed.utils import is_distributed_ready
from gravitorch.engines.base import BaseEngine
from gravitorch.handlers.base import BaseHandler
from gravitorch.runners.base import BaseRunner
from gravitorch.runners.utils import configure_pytorch
from gravitorch.utils.exp_trackers import BaseExpTracker
from gravitorch.utils.format import str_add_indent, to_pretty_json_str
from gravitorch.utils.logging import disable_logging

logger = logging.getLogger(__name__)


class BaseDistributedRunner(BaseRunner):
    r"""Defines a base class to easily implement a distributed runner.

    Note that this class provides a default implementation that covers
    a large set of use cases, but you can use other implementations
    if it does not fit with your use case.

    If you define a child class, you should only define the ``_run()``
    method. The ``run()`` method is already implemented.

    Args:
        dist_backend (str or None, optional): Specifies the
            distributed backend. The possible values are:
                - ``'auto'``: automatically find the best distributed
                    backend.
                - ``'nccl'``: uses the NCCL backend.
                - ``'gloo'``: uses the GLOO backend.
                - ``None``: does not use distributed backend.
                Default: ``'auto'``
        log_only_main_process (bool, optional): If ``True``, only the
            outputs of the main process are logged. The logging of
            other processes is limited to the error level or above.
            If ``False``, the outputs of all the processes are logged.
            Default: ``True``
    """

    def __init__(self, dist_backend: Optional[str] = "auto", log_only_main_process: bool = True):
        self._dist_backend = resolve_dist_backend(dist_backend)
        logger.info(f"distributed backend: {self._dist_backend}")
        self._log_only_main_process = log_only_main_process

    def run(self) -> Any:
        r"""Sets up the distributed context and executes the logic of the
        runner.

        Returns:
            Any artifact of the runner
        """
        with auto_distributed_context(self._dist_backend):
            with self._setup_logging_context():
                return self._run()

    @abstractmethod
    def _run(self) -> Any:
        r"""Executes the logic of the runner after the .

        Returns:
            Any artifact of the runner
        """

    def _setup_logging_context(self) -> AbstractContextManager:
        r"""Sets up the logging context.

        The main use of this function is to control the logging, in
        particular in a distributed setting. The current
        implementation disables the logging for the non-main
        processes.

        Returns:
            ``contextlib.AbstractContextManager``: The context manager
                that controls the logging.
        """
        if self._log_only_main_process and not dist.is_main_process():
            return disable_logging(logging.ERROR - 1)
        return nullcontext()


def resolve_dist_backend(dist_backend: Optional[str]) -> Optional[str]:
    r"""Resolves the distributed backend if ``'auto'``.

    Args:
        dist_backend (str or ``None``): Specifies the distributed
            backend. If ``'auto'``, this function will find the best
            option for the distributed backend according to the
            context and some rules.

    Returns:
        str or ``None``: The distributed backend or ``None`` if it
            should not use a distributed backend.
    """
    if dist_backend == "auto":
        if is_distributed_ready():
            dist_backend = auto_dist_backend()
        else:
            # Set to ``None`` because the process does not seem ready
            # to be configured for a distributed experiment.
            dist_backend = None
    return dist_backend


class BaseEngineDistributedRunner(BaseDistributedRunner):
    r"""Implements a runner with an engine.

    Args:
        engine (``BaseEngine`` or dict): Specifies the engine or its
            configuration.
        handlers (list or tuple or ``None``): Specifies the list of
            handlers or their configuration. The handlers will be
            attached to the engine. If ``None``, no handler is
            attached to the engine. Default: ``None``
        exp_tracker (``BaseExpTracker`` or dict or None): Specifies
            the experiment tracker or its configuration. If ``None``,
            the no-operation experiment tracker is used.
        random_seed (int, optional): Specifies the random seed.
            Default: ``10139531598155730726``
        dist_backend (str or None, optional): Specifies the
            distributed backend. The possible values are:
                - ``'auto'``: automatically find the best distributed
                    backend.
                - ``'nccl'``: uses the NCCL backend.
                - ``'gloo'``: uses the GLOO backend.
                - ``None``: does not use distributed backend.
            Default: ``'auto'``
        log_only_main_process (bool, optional): If ``True``, only
            the outputs of the main process are logged. The logging
            of other processes is limited to the error level or above.
            If ``False``, the outputs of all the processes are logged.
            Default: ``True``
        pytorch_config (dict or None, optional): Specifies some
            PyTorch options. Default: ``None``
    """

    def __init__(
        self,
        engine: Union[BaseEngine, dict],
        handlers: Union[
            tuple[Union[BaseHandler, dict], ...], list[Union[BaseHandler, dict]], None
        ] = None,
        exp_tracker: Union[BaseExpTracker, dict, None] = None,
        random_seed: int = 10139531598155730726,
        dist_backend: Optional[str] = "auto",
        log_only_main_process: bool = True,
        pytorch_config: Optional[dict] = None,
    ):
        super().__init__(dist_backend=dist_backend, log_only_main_process=log_only_main_process)
        self._engine = engine
        self._handlers = handlers or tuple()
        self._exp_tracker = exp_tracker
        self._random_seed = random_seed
        self._pytorch_config = pytorch_config or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  engine={str_add_indent(to_pretty_json_str(self._engine))},\n"
            f"  handlers={str_add_indent(to_pretty_json_str(self._handlers))},\n"
            f"  exp_tracker={str_add_indent(to_pretty_json_str(self._exp_tracker))},\n"
            f"  random_seed={self._random_seed},\n"
            f"  dist_backend={self._dist_backend},\n"
            f"  log_only_main_process={self._log_only_main_process},\n"
            f"  pytorch_config={str_add_indent(to_pretty_json_str(self._pytorch_config))},\n"
            ")"
        )

    def run(self) -> Any:
        r"""Sets up the distributed context and executes the logic of the
        runner.

        Returns:
            Any artifact of the runner
        """
        with auto_distributed_context(self._dist_backend):
            with self._setup_logging_context():
                configure_pytorch(**self._pytorch_config)
                return self._run()
