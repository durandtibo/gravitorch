__all__ = ["BaseResourceRunner"]

import logging
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any, Optional, Union

from gravitorch.rsrc.base import BaseResource, setup_resource
from gravitorch.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class BaseResourceRunner(BaseRunner):
    r"""Implements a base class to easily implement a runner using resource
    managers.

    Args:
        resources (sequence or ``None``, optional): Specifies a
            sequence of resources or their configurations.
            Default: ``None``
    """

    def __init__(self, resources: Optional[Sequence[Union[BaseResource, dict]]] = None):
        if resources is None:
            resources = tuple()
        self._resources = tuple(setup_resource(resource) for resource in resources)

    def run(self) -> Any:
        r"""Sets up the distributed context and executes the logic of the
        runner.

        Returns:
            Any artifact of the runner
        """
        with ExitStack() as stack:
            for resource in self._resources:
                stack.enter_context(resource)
            output = self._run()
        return output

    @abstractmethod
    def _run(self) -> Any:
        r"""Executes the logic of the runner after the .

        Returns:
            Any artifact of the runner
        """
