from __future__ import annotations

__all__ = ["DataLoaderDataFlow"]

from collections.abc import Iterator

from torch.utils.data import DataLoader

from gravitorch.experimental.dataflow.base import BaseDataFlow
from gravitorch.utils.imports import is_torchdata_available

DataLoaderTypes = (DataLoader,)
if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2

    DataLoaderTypes = (DataLoader, DataLoader2)
else:
    DataLoader2 = "DataLoader2"  # pragma: no cover


class DataLoaderDataFlow(BaseDataFlow):
    r"""Implements a simple dataflow for PyTorch data loaders.

    Args:
        dataloader (``DataLoader`` or ``DataLoader2``): Specifies the
            data loader.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> from gravitorch.experimental.dataflow import IterableDataFlow
        >>> dataloader = DataLoader(TensorDataset(torch.arange(10)), batch_size=4)
        >>> with DataLoaderDataFlow(dataloader) as dataflow:
        ...     for batch in dataflow:
        ...         print(batch)  # do something
        ...
    """

    def __init__(self, dataloader: DataLoader | DataLoader2) -> None:
        if not isinstance(dataloader, DataLoaderTypes):
            raise TypeError(
                "Incorrect type. Expecting DataLoader or DataLoader2 but "
                f"received {type(dataloader)}"
            )
        self.dataloader = dataloader

    def __iter__(self) -> Iterator:
        return iter(self.dataloader)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def launch(self) -> None:
        r"""Nothing to do for this dataflow."""

    def shutdown(self) -> None:
        if isinstance(self.dataloader, DataLoader2):
            self.dataloader.shutdown()
