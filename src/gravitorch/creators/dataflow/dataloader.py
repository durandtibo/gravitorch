from __future__ import annotations

__all__ = ["DataLoaderDataFlowCreator"]

from typing import TYPE_CHECKING, TypeVar

from coola.utils import str_indent, str_mapping
from torch.utils.data import DataLoader

from gravitorch.creators.dataflow.base import BaseDataFlowCreator
from gravitorch.data.dataloaders.factory import is_dataloader_config
from gravitorch.dataflows.dataloader import DataLoaderDataFlow
from gravitorch.experimental.dataloader.base import (
    BaseDataLoaderCreator,
    setup_dataloader_creator,
)
from gravitorch.experimental.dataloader.vanilla import DataLoaderCreator

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class DataLoaderDataFlowCreator(BaseDataFlowCreator[T]):
    r"""Implements a simple ``DataLoaderDataFlow`` creator.

    Args:
    ----
        dataloader (``torch.utils.data.DataLoader`` or
            ``BaseDataLoaderCreator``): Specifies a dataloader (or its
            configuration) or a dataloader creator (or its
            configuration).

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import ExampleDataset
        >>> from gravitorch.creators.dataflow import DataLoaderDataFlowCreator
        >>> from torch.utils.data import DataLoader
        >>> creator = DataLoaderDataFlowCreator(DataLoader(ExampleDataset([1, 2, 3, 4, 5])))
        >>> creator
        DataLoaderDataFlowCreator(
          (dataloader): DataLoaderCreator(
              cache=False
              dataloader=<torch.utils.data.dataloader.DataLoader object at 0x...>
            )
        )
        >>> dataflows = creator.create()
        >>> dataflows
        DataLoaderDataFlow(length=5)
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

    def create(self, engine: BaseEngine | None = None) -> DataLoaderDataFlow[T]:
        return DataLoaderDataFlow(self._dataloader.create(engine))
