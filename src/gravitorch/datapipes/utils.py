from __future__ import annotations

__all__ = ["clone_datapipe"]

import copy
import logging

from torch.utils.data.graph import DataPipe

logger = logging.getLogger(__name__)


def clone_datapipe(datapipe: DataPipe, raise_error: bool = True) -> DataPipe:
    r"""Clone (a.k.a. deepcopy) the ``DataPipe``.

    Args:
    ----
        datapipe (``DataPipe``): Specifies the ``DataPipe`` to clone.
        raise_error (bool, optional): If ``True``, this function will
            raise an error if the ``DataPipe`` cannot be cloned.
            If ``False``, the original ``DataPipe`` is returned if the
            ``DataPipe`` cannot be cloned. Default: ``True``

    Returns:
    -------
        ``DataPipe``: The cloned ``DataPipe``.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import clone_datapipe
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> clone_datapipe(IterableWrapper([1, 2, 1, 3]))
        IterableWrapperIterDataPipe
    """
    try:
        datapipe = copy.deepcopy(datapipe)
    except TypeError as exc:
        if raise_error:
            raise TypeError("The DataPipe can not be cloned") from exc
        logger.warning(
            "The DataPipe can not be cloned, please be aware of in-place "
            "modification would affect source DataPipe."
        )
    return datapipe
