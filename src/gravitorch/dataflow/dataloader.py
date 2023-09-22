from __future__ import annotations

__all__ = ["DataLoaderDataFlow"]

from collections.abc import Iterator

from torch.utils.data import DataLoader

from gravitorch.dataflow.base import BaseDataFlow
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2
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
        >>> from gravitorch.dataflow import IterableDataFlow
        >>> dataloader = DataLoader(TensorDataset(torch.arange(10)), batch_size=4)
        >>> with DataLoaderDataFlow(dataloader) as dataflow:
        ...     for batch in dataflow:
        ...         print(batch)  # do something
        ...
    """

    def __init__(self, dataloader: DataLoader | DataLoader2) -> None:
        if not isinstance(
            dataloader, (DataLoader, DataLoader2) if is_torchdata_available() else DataLoader
        ):
            raise TypeError(
                "Incorrect type. Expecting DataLoader or DataLoader2 but "
                f"received {type(dataloader)}"
            )
        self.dataloader = dataloader  # TODO: add support for config

    def __iter__(self) -> Iterator:
        return iter(self.dataloader)

    def __repr__(self) -> str:
        try:
            extra = f"length={len(self.dataloader):,}"
        except TypeError:
            extra = ""
        return f"{self.__class__.__qualname__}({extra})"

    def launch(self) -> None:
        r"""Nothing to do for this dataflow."""

    def shutdown(self) -> None:
        if is_torchdata_available() and isinstance(self.dataloader, DataLoader2):
            self.dataloader.shutdown()
