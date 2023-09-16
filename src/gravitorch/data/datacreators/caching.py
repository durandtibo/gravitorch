from __future__ import annotations

__all__ = ["CacheDataCreator"]

import copy
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from coola.utils import str_indent, str_mapping

from gravitorch.data.datacreators.base import BaseDataCreator, setup_datacreator

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheDataCreator(BaseDataCreator[T]):
    r"""Implements a data creator that creates the data only once, then
    cache them and return them.

    Args:
    ----
        creator (``BaseDataCreator`` or dict): Specifies a data
            creator or its configuration.
        deepcopy (bool, optional): If ``True``, a deepcopy of the data
            is done before to return the data. If ``deepcopy`` is
            explicitly set to ``False``, users should ensure that
            the data pipeline does not contain any in-place operations
            over the data to prevent data inconsistency if the
            ``create`` method is called multiple times.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.datacreators import CacheDataCreator, HypercubeVertexDataCreator
        >>> creator = CacheDataCreator(
        ...     HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=6)
        ... )
        >>> creator
        CacheDataCreator(
          (creator): HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=6, noise_std=0.2, random_seed=15782179921860610490)
          (is_cache_created): False
          (deepcopy): False
        )
        >>> data = creator.create()
        >>> data
        {'target': tensor([...]), 'input': tensor([[...]])}
    """

    def __init__(self, creator: BaseDataCreator[T] | dict, deepcopy: bool = False) -> None:
        self._creator = setup_datacreator(creator)
        self._deepcopy = bool(deepcopy)
        self._is_cache_created = False
        # This variable is used to cache the data. The type depends on the value returned by
        # the function ``create`` of the data creator object.
        self._cached_data: Any = None

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "creator": self._creator,
                    "is_cache_created": self._is_cache_created,
                    "deepcopy": self._deepcopy,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def creator(self) -> BaseDataCreator[T]:
        r"""``BaseDataCreator``: The data creator."""
        return self._creator

    @property
    def deepcopy(self) -> bool:
        r"""``bool``: Indicates if a deepcopy of the data is done before
        to return the data."""
        return self._deepcopy

    def create(self, engine: BaseEngine | None = None) -> T:
        r"""Creates data.

        Args:
        ----
            engine (``BaseEngine`` or ``None``): Specifies an engine.
                Default: ``None``

        Return:
        ------
            The created data. The returned data depends on the data
                creator.
        """
        if not self._is_cache_created:
            logger.info("Creating data and caching them...")
            self._cached_data = self._creator.create(engine)
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
