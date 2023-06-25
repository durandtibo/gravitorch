from __future__ import annotations

__all__ = ["create_dataloader2"]

from collections.abc import Iterable

from torch.utils.data.graph import DataPipe

from gravitorch.utils import setup_object
from gravitorch.utils.imports import check_torchdata, is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2, ReadingServiceInterface
    from torchdata.dataloader2.adapter import Adapter
else:  # pragma: no cover
    Adapter = "Adapter"
    DataLoader2 = "DataLoader2"
    ReadingServiceInterface = "ReadingServiceInterface"


def create_dataloader2(
    datapipe: DataPipe | dict,
    datapipe_adapter_fn: Iterable[Adapter | dict] | Adapter | dict | None = None,
    reading_service: ReadingServiceInterface | dict | None = None,
) -> DataLoader2:
    r"""Creates a ``torchdata.dataloader2.DataLoader2`` object from a DataPipe or its configuration.

    Args:
        datapipe: Specifies the DataPipe or its configuration.
        datapipe_adapter_fn: Specifies the ``Adapter`` function(s) that
            will be applied to the DataPipe. Default: ``None``
        reading_service: Defines how ``DataLoader2`` should execute
            operations over the ``DataPipe``, e.g.
            multiprocessing/distributed. Default: ``None``

    Returns:
        ``torchdata.dataloader2.DataLoader2``: The instantiated
            ``DataLoader2``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.dataloaders import create_dataloader2
        >>> create_dataloader2(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": range(10),
        ...     }
        ... )
        <torchdata.dataloader2.dataloader2.DataLoader2 at 0x0123456789>
    """
    check_torchdata()
    return DataLoader2(
        datapipe=setup_object(datapipe),
        datapipe_adapter_fn=setup_object(datapipe_adapter_fn),
        reading_service=setup_object(reading_service),
    )
