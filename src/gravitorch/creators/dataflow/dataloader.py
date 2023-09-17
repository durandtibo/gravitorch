from __future__ import annotations

__all__ = ["DataLoaderDataFlowCreator"]

from typing import TypeVar

from coola.utils import str_indent, str_mapping
from torch.utils.data import DataLoader

from gravitorch.creators.dataflow.base import BaseDataFlowCreator
from gravitorch.creators.dataloader import DataLoaderCreator, setup_dataloader_creator
from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.data.dataloaders import is_dataloader_config
from gravitorch.dataflow.dataloader import DataLoaderDataFlow
from gravitorch.engines.base import BaseEngine

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

        >>> from gravitorch.data.datasets import ExampleDataset
        >>> from gravitorch.creators.dataflow import DataLoaderDataFlowCreator
        >>> from torch.utils.data import DataLoader
        >>> creator = DataLoaderDataFlowCreator(DataLoader(ExampleDataset([1, 2, 3, 4, 5])))
        >>> dataloader = creator.create()
        >>> dataloader
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
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(config, sorted_keys=True))}\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoaderDataFlow[T]:
        return DataLoaderDataFlow(self._dataloader.create(engine))
