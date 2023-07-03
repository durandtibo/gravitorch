from __future__ import annotations

__all__ = ["setup_datapipe"]

import logging

from objectory import factory
from torch.utils.data.graph import DataPipe

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_datapipe(datapipe: DataPipe | dict) -> DataPipe:
    r"""Sets up a ``torch.utils.data.graph.DataPipe`` object.

    Args:
    ----
        datapipe (``torch.utils.data.graph.DataPipe`` or dict):
            Specifies the datapipe or its configuration (dictionary).

    Returns:
    -------
        ``torch.utils.data.graph.DataPipe``: The instantiated
            ``torch.utils.data.graph.DataPipe`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import setup_datapipe
        >>> datapipe = setup_datapipe(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> datapipe
        IterableWrapperIterDataPipe
        >>> setup_datapipe(datapipe)  # Do nothing because the datapipe is already instantiated
        IterableWrapperIterDataPipe
    """
    if isinstance(datapipe, dict):
        logger.info(
            "Initializing a `torch.utils.data.graph.DataPipe` from its configuration... "
            f"{str_target_object(datapipe)}"
        )
        datapipe = factory(**datapipe)
    return datapipe
