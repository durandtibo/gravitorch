from __future__ import annotations

__all__ = ["SequentialRunner"]

import logging
from collections.abc import Sequence
from typing import Any

from coola.utils import str_indent, str_sequence

from gravitorch.runners.base import BaseRunner
from gravitorch.runners.utils import setup_runner

logger = logging.getLogger(__name__)


class SequentialRunner(BaseRunner):
    r"""Implements a runner that executes multiple runners sequentially.

    Args:
    ----
        runners (sequence): Specifies the sequence of runners or their
            configuration.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners import TrainingRunner, SequentialRunner, EvaluationRunner
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> runner = SequentialRunner([TrainingRunner(engine), EvaluationRunner(engine)])
        >>> runner.run()
    """

    def __init__(self, runners: Sequence[BaseRunner | dict]) -> None:
        self._runners = tuple(setup_runner(runner) for runner in runners)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._runners))}\n)"

    def run(self) -> Any:
        for runner in self._runners:
            runner.run()
