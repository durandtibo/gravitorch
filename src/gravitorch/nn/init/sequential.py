__all__ = ["SequentialInitializer"]

import logging
from collections.abc import Sequence
from typing import Union

from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer
from gravitorch.nn.init.factory import setup_initializer
from gravitorch.utils.format import str_indent, to_torch_sequence_str

logger = logging.getLogger(__name__)


class SequentialInitializer(BaseInitializer):
    r"""Implements a module initializer that sequentially calls module
    initializers.

    Args:
    ----
        initializers: Specifies the sequence of module initializers.
            The sequence order defines the order of the call.
    """

    def __init__(
        self,
        initializers: Sequence[Union[BaseInitializer, dict]],
    ) -> None:
        super().__init__()
        self._initializers = tuple(setup_initializer(initializer) for initializer in initializers)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(to_torch_sequence_str(self._initializers))}\n)"
        )

    def initialize(self, module: Module) -> None:
        for i, initializer in enumerate(self._initializers):
            logger.info(f"[{i}/{len(self._initializers)}] {initializer}")
            initializer.initialize(module)
