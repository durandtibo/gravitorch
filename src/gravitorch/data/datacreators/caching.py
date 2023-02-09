__all__ = ["OneCacheDataCreator"]

import copy
import logging
from typing import Any, Optional, TypeVar, Union

from gravitorch.data.datacreators import BaseDataCreator, setup_data_creator
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OneCacheDataCreator(BaseDataCreator[T]):
    r"""Implements a data creator that creates the data only once, then cache
    them and return them.

    Args:
        data_creator (``BaseDataCreator`` or dict): Specifies a data
            creator or its configuration.
        deepcopy (bool, optional): If ``True``, a deepcopy of the data
            is done before to return the data. If ``deepcopy`` is
            explicitly set to ``False``, users should ensure that
            the data pipeline does not contain any in-place operations
            over the data to prevent data inconsistency if the
            ``create`` method is called multiple times.
    """

    def __init__(self, data_creator: Union[BaseDataCreator[T], dict], deepcopy: bool = False):
        self._data_creator = setup_data_creator(data_creator)
        self._deepcopy = bool(deepcopy)
        self._is_cache_created = False
        # This variable is used to cache the data. The type depends on the value returned by
        # the function ``create`` of the data creator object.
        self._cached_data: Any = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  data_creator={str_indent(self._data_creator)},\n"
            f"  is_cache_created={self._is_cache_created},\n"
            f"  deepcopy={self._deepcopy},\n"
            ")"
        )

    @property
    def data_creator(self) -> BaseDataCreator[T]:
        r"""``BaseDataCreator``: The data creator."""
        return self._data_creator

    @property
    def deepcopy(self) -> bool:
        r"""bool: Indicates if a deepcopy of the data is done before
        to return the data."""
        return self._deepcopy

    def create(self, engine: Optional[BaseEngine] = None) -> T:
        r"""Creates data.

        Args:
            engine (``BaseEngine`` or ``None``): Specifies an engine.
            Default: ``None``

        Return:
            The created data. The returned data depends on the data
                creator.
        """
        if not self._is_cache_created:
            logger.info("Creating data and caching them...")
            self._cached_data = self._data_creator.create(engine)
            self._is_cache_created = True
        data = self._cached_data
        if self._deepcopy:
            try:
                data = copy.deepcopy(data)
            except TypeError:
                logger.warning(
                    "The data can not be deepcopied. "
                    "Please be aware of in-place modification would affect source data"
                )
        return data
