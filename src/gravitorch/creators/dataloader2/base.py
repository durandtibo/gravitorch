from __future__ import annotations

__all__ = ["BaseDataLoader2Creator", "is_dataloader2_creator_config", "setup_dataloader2_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

from gravitorch.utils.format import str_target_object
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2
else:  # pragma: no cover
    DataLoader2 = "DataLoader2"

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataLoader2Creator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a dataloader creator.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.creators.dataloader2 import VanillaDataLoader2Creator
        >>> creator = VanillaDataLoader2Creator(IterableWrapper([1, 2, 3, 4, 5]))
        >>> creator.create()
        <torchdata.dataloader2.dataloader2.DataLoader2 object at 0x...>
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> DataLoader2[T]:
        r"""Create a dataloader.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``torchdata.dataloader2.DataLoader2``: The created dataloader.
        """


def is_dataloader2_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDataLoader2Creator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseDataLoader2Creator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataloader2 import is_dataloader2_creator_config
        >>> is_dataloader2_creator_config(
        ...     {"_target_": "gravitorch.creators.dataloader2.DataLoader2Creator"}
        ... )
        True
    """
    return is_object_config(config, BaseDataLoader2Creator)


def setup_dataloader2_creator(
    creator: BaseDataLoader2Creator[T] | dict,
) -> BaseDataLoader2Creator[T]:
    r"""Sets up the dataloader creator.

    The dataloader creator is instantiated from its configuration by
    using the ``BaseDataLoader2Creator`` factory function.

    Args:
    ----
        creator (``BaseDataLoader2Creator`` or dict): Specifies the
            dataloader creator or its configuration.

    Returns:
    -------
        ``BaseDataLoader2Creator``: The instantiated dataloader creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataloader2 import setup_dataloader2_creator
        >>> creator = setup_dataloader2_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.dataloader2.VanillaDataLoader2Creator",
        ...         "datapipe": {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4, 5],
        ...         },
        ...     }
        ... )
        >>> creator
        VanillaDataLoader2Creator(
          (datapipe): DataPipeCreator(
              cache=False
              datapipe={'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4, 5]}
              deepcopy=False
            )
          (datapipe_adapter_fn): None
          (reading_service): None
        )
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the dataloader creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataLoader2Creator.factory(**creator)
    return creator
