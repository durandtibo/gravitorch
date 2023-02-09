__all__ = ["SequentialParameterInitializer"]

import logging
from collections.abc import Sequence
from typing import Union

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent, to_torch_sequence_str
from gravitorch.utils.parameter_initializers.base import (
    BaseDefaultParameterInitializer,
    BaseParameterInitializer,
)
from gravitorch.utils.parameter_initializers.utils import setup_parameter_initializer

logger = logging.getLogger(__name__)


class SequentialParameterInitializer(BaseDefaultParameterInitializer):
    r"""Implements a parameter initializer that sequentially calls parameter
    initializer.

    Args:
        parameter_initializers: Specifies the sequence of parameter
            initializers. The sequence order defines the order of the
            call.
        show_stats (bool, optional): If ``True``, the parameter
            statistics are shown at the end of the initialization.
            Default: ``True``
    """

    def __init__(
        self,
        parameter_initializers: Sequence[Union[BaseParameterInitializer, dict]],
        show_stats: bool = True,
    ):
        super().__init__(show_stats=show_stats)
        self._parameter_initializers = tuple(
            setup_parameter_initializer(param_init) for param_init in parameter_initializers
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(to_torch_sequence_str(self._parameter_initializers))}\n)"
        )

    def _initialize(self, engine: BaseEngine) -> None:
        for i, parameter_initializer in enumerate(self._parameter_initializers):
            logger.info(f"[{i}/{len(self._parameter_initializers)}] {parameter_initializer}")
            parameter_initializer.initialize(engine)
