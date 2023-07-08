from __future__ import annotations

__all__ = ["DataCreator"]

import copy
import logging
from typing import TypeVar

from gravitorch.data.datacreators.base import BaseDataCreator
from gravitorch.engines.base import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DataCreator(BaseDataCreator[T]):
    r"""Implement a simple data creator.

    Args:
    ----
        data: Specifies the data to create.
        deepcopy (bool, optional): If ``True``, the data is deep-copied
            before to iterate over the data. It allows a deterministic
            behavior when in-place operations are performed on the
            data. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.datacreators import DataCreator
        >>> creator = DataCreator([1, 2, 3, 4])
        >>> creator.create()
        [1, 2, 3, 4]
    """

    def __init__(self, data: T, deepcopy: bool = False) -> None:
        self._data = data
        self._deepcopy = bool(deepcopy)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(deepcopy={self._deepcopy})"

    def create(self, engine: BaseEngine | None = None) -> T:
        data = self._data
        if self._deepcopy:
            try:
                data = copy.deepcopy(data)
            except TypeError:
                logger.warning(
                    "The data can not be deepcopied, please be aware of in-place "
                    "modification would affect the source data."
                )
        return data
