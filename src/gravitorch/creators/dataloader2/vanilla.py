from __future__ import annotations

__all__ = ["DataLoader2Creator", "VanillaDataLoader2Creator"]

import logging
from collections.abc import Iterable
from typing import TypeVar

from coola.utils import str_indent, str_mapping
from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.creators.dataloader2.base import BaseDataLoader2Creator
from gravitorch.creators.datapipe.base import (
    BaseDataPipeCreator,
    setup_datapipe_creator,
)
from gravitorch.creators.datapipe.vanilla import DataPipeCreator
from gravitorch.data.dataloaders import create_dataloader2, setup_dataloader2
from gravitorch.datapipes import is_datapipe_config
from gravitorch.engines import BaseEngine
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2, ReadingServiceInterface
    from torchdata.dataloader2.adapter import Adapter
else:  # pragma: no cover
    Adapter = "Adapter"
    DataLoader2 = "DataLoader2"
    ReadingServiceInterface = "ReadingServiceInterface"

T = TypeVar("T")

logger = logging.getLogger(__name__)


class DataLoader2Creator(BaseDataLoader2Creator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        dataloader (``torchdata.dataloader2.DataLoader2`` or dict):
            Specifies the dataloader or its configuration.
        cache (bool, optional): If ``True``, the dataloader is created
            only the first time, and then the same data is returned
            for each call to the ``create`` method.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataloader2 import DataLoader2Creator
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from torchdata.dataloader2 import DataLoader2
        >>> creator = DataLoader2Creator(
        ...     {
        ...         "_target_": "torchdata.dataloader2.DataLoader2",
        ...         "datapipe": IterableWrapper((1, 2, 3, 4)),
        ...     },
        ... )
        >>> creator.create()
        <torchdata.dataloader2.dataloader2.DataLoader2 object at 0x...>
    """

    def __init__(self, dataloader: DataLoader2 | dict, cache: bool = False) -> None:
        self._dataloader = dataloader
        self._cache = bool(cache)

    def __repr__(self) -> str:
        config = {"dataloader": self._dataloader, "cache": self._cache}
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(config, sorted_keys=True))}\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoader2[T]:
        dataloader = setup_dataloader2(self._dataloader)
        if self._cache:
            self._dataloader = dataloader
        return dataloader


class VanillaDataLoader2Creator(BaseDataLoader2Creator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        datapipe (``IterDataPipe`` or ``MapDataPipe`` or
            ``BaseDataPipeCreator`` or dict): Specifies a
            datapipe (or its configuration) or a datapipe creator
            (or its configuration).
        datapipe_adapter_fn: Specifies the ``Adapter`` function(s)
            that will be applied to the DataPipe. Default: ``None``
        reading_service: Defines how ``DataLoader2`` should execute
            operations over the ``DataPipe``, e.g.
            multiprocessing/distributed. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.creators.dataloader2 import VanillaDataLoader2Creator
        >>> creator = VanillaDataLoader2Creator(IterableWrapper([1, 2, 3, 4, 5]))
        >>> creator.create()
        <torchdata.dataloader2.dataloader2.DataLoader2 object at 0x...>
    """

    def __init__(
        self,
        datapipe: IterDataPipe[T] | MapDataPipe[T] | BaseDataPipeCreator[T] | dict,
        datapipe_adapter_fn: Iterable[Adapter | dict] | Adapter | dict | None = None,
        reading_service: ReadingServiceInterface | dict | None = None,
    ) -> None:
        if isinstance(datapipe, (IterDataPipe, MapDataPipe)) or (
            isinstance(datapipe, dict) and is_datapipe_config(datapipe)
        ):
            datapipe = DataPipeCreator(datapipe)
        self._datapipe = setup_datapipe_creator(datapipe)
        self._datapipe_adapter_fn = datapipe_adapter_fn
        self._reading_service = reading_service

    def __repr__(self) -> str:
        config = {
            "datapipe": self._datapipe,
            "datapipe_adapter_fn": self._datapipe_adapter_fn,
            "reading_service": self._reading_service,
        }
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(config))}\n)"

    def create(self, engine: BaseEngine | None = None) -> DataLoader2[T]:
        datapipe = self._datapipe.create(engine)
        logger.info(f"datapipe:\n{datapipe}")
        return create_dataloader2(
            datapipe=datapipe,
            datapipe_adapter_fn=self._datapipe_adapter_fn,
            reading_service=self._reading_service,
        )
