from __future__ import annotations

__all__ = ["VanillaDataSource"]

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from coola.utils import str_indent, str_mapping

from gravitorch.creators.datastream.base import setup_datastream_creator
from gravitorch.datasources.base import BaseDataSource, IterableNotFoundError
from gravitorch.datastreams.base import BaseDataStream
from gravitorch.utils.asset import AssetManager

if TYPE_CHECKING:
    from gravitorch.creators.datastream import BaseDataStreamCreator
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VanillaDataSource(BaseDataSource):
    r"""Implement a simple data source using datastream creators.

    Args:
    ----
        datastream_creators (``Mapping``): Specifies the datastreams
            creators or their configuration.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasources import VanillaDataSource
        >>> from gravitorch.creators.datastream import IterableDataStreamCreator
        >>> datasource = VanillaDataSource(
        ...     {
        ...         "train": {
        ...             "_target_": "gravitorch.creators.datastream.IterableDataStreamCreator",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...         "eval": IterableDataStreamCreator(["a", "b", "c"]),
        ...     }
        ... )
        >>> datasource
        VanillaDataSource(
          (train): IterableDataStreamCreator(cache=False, length=4)
          (eval): IterableDataStreamCreator(cache=False, length=3)
        )
    """

    def __init__(self, datastream_creators: Mapping[str, BaseDataStreamCreator[T] | dict]) -> None:
        self._asset_manager = AssetManager()
        self._datastream_creators: Mapping[str, BaseDataStreamCreator] = {
            key: setup_datastream_creator(value) for key, value in datastream_creators.items()
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(self._datastream_creators))}\n)"
        )

    def attach(self, engine: BaseEngine) -> None:
        logger.info("Attach the data source to an engine")

    def get_asset(self, asset_id: str) -> Any:
        return self._asset_manager.get_asset(asset_id)

    def has_asset(self, asset_id: str) -> bool:
        return self._asset_manager.has_asset(asset_id)

    def get_iterable(self, iter_id: str, engine: BaseEngine | None = None) -> BaseDataStream[T]:
        if not self.has_iterable(iter_id):
            raise IterableNotFoundError(f"{iter_id} does not exist")
        return self._datastream_creators[iter_id].create(engine=engine)

    def has_iterable(self, iter_id: str) -> bool:
        return iter_id in self._datastream_creators
