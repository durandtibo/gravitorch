from __future__ import annotations

__all__ = ["AutoDataLoaderCreator"]

from typing import TYPE_CHECKING, TypeVar

from torch.utils.data import DataLoader, Dataset

from gravitorch.creators.dataset.base import BaseDatasetCreator
from gravitorch.distributed import comm as dist
from gravitorch.experimental.dataloader.base import BaseDataLoaderCreator
from gravitorch.experimental.dataloader.distributed import DistributedDataLoaderCreator
from gravitorch.experimental.dataloader.vanilla import VanillaDataLoaderCreator
from gravitorch.utils.format import str_indent

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class AutoDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a PyTorch dataloader creator that automatically chooses
    the dataloader creator based on the context.

    If the distributed package is activated, it uses the
    ``DistributedDataLoaderCreator``, otherwise it uses
    ``DataLoaderCreator``.

    Note the behavior of this class may change based on the new data
    loader creators.

    Args:
    ----
        dataset (``torch.utils.data.Dataset``): Specifies a
            dataset (or its configuration) or a dataset creator
            (or its configuration).
        **kwargs: See ``DataLoaderCreator`` or
            ``DistributedDataLoaderCreator`` documentation.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.experimental.dataloader import AutoDataLoaderCreator
        >>> creator = AutoDataLoaderCreator(
        ...     {
        ...         "_target_": "gravitorch.datasets.DummyMultiClassDataset",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> creator.create()
        <torch.utils.data.dataloader.DataLoader object at 0x...>
    """

    def __init__(self, dataset: Dataset | BaseDatasetCreator | dict, **kwargs) -> None:
        if dist.is_distributed():
            self._dataloader = DistributedDataLoaderCreator(dataset, **kwargs)
        else:
            self._dataloader = VanillaDataLoaderCreator(dataset, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  dataloader={str_indent(self._dataloader)}\n)"

    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        return self._dataloader.create(engine=engine)
