from __future__ import annotations

__all__ = ["DataLoaderDataStreamCreator"]

from typing import TYPE_CHECKING, TypeVar

from coola.utils import str_indent, str_mapping
from torch.utils.data import DataLoader

from gravitorch.creators.datastream.base import BaseDataStreamCreator
from gravitorch.dataloaders.factory import is_dataloader_config
from gravitorch.datastreams.dataloader import DataLoaderDataStream
from gravitorch.experimental.dataloader.base import (
    BaseDataLoaderCreator,
    setup_dataloader_creator,
)
from gravitorch.experimental.dataloader.vanilla import DataLoaderCreator

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class DataLoaderDataStreamCreator(BaseDataStreamCreator[T]):
    r"""Implements a simple ``DataLoaderDataStream`` creator.

    Args:
    ----
        dataloader (``torch.utils.data.DataLoader`` or
            ``BaseDataLoaderCreator``): Specifies a dataloader (or its
            configuration) or a dataloader creator (or its
            configuration).

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import ExampleDataset
        >>> from gravitorch.creators.datastream import DataLoaderDataStreamCreator
        >>> from torch.utils.data import DataLoader
        >>> creator = DataLoaderDataStreamCreator(DataLoader(ExampleDataset([1, 2, 3, 4, 5])))
        >>> creator
        DataLoaderDataStreamCreator(
          (dataloader): DataLoaderCreator(
              cache=False
              dataloader=<torch.utils.data.dataloader.DataLoader object at 0x...>
            )
        )
        >>> datastream = creator.create()
        >>> datastream
        DataLoaderDataStream(length=5)
    """

    def __init__(self, dataloader: DataLoader | BaseDataLoaderCreator | dict) -> None:
        if isinstance(dataloader, DataLoader) or (
            isinstance(dataloader, dict) and is_dataloader_config(dataloader)
        ):
            dataloader = DataLoaderCreator(dataloader)
        self._dataloader = setup_dataloader_creator(dataloader)

    def __repr__(self) -> str:
        config = {"dataloader": self._dataloader}
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(config))}\n)"

    def create(self, engine: BaseEngine | None = None) -> DataLoaderDataStream[T]:
        return DataLoaderDataStream(self._dataloader.create(engine))
