from __future__ import annotations

__all__ = ["BaseDataLoaderCreator", "is_dataloader_creator_config", "setup_dataloader_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config
from torch.utils.data import DataLoader

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataLoaderCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a dataloader creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.experimental.dataloader import VanillaDataLoaderCreator
        >>> creator = VanillaDataLoaderCreator(
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

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        r"""Create a dataloader.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``torch.utils.data.DataLoader``: The created dataloader.
        """


def is_dataloader_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDataLoaderCreator``.

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
            for a ``BaseDataLoaderCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.experimental.dataloader import is_dataloader_creator_config
        >>> is_dataloader_creator_config(
        ...     {"_target_": "gravitorch.experimental.dataloader.DataLoaderCreator"}
        ... )
        True
    """
    return is_object_config(config, BaseDataLoaderCreator)


def setup_dataloader_creator(creator: BaseDataLoaderCreator[T] | dict) -> BaseDataLoaderCreator[T]:
    r"""Sets up the dataloader creator.

    The dataloader creator is instantiated from its configuration by
    using the ``BaseDataLoaderCreator`` factory function.

    Args:
    ----
        creator (``BaseDataLoaderCreator`` or dict): Specifies the
            dataloader creator or its configuration.

    Returns:
    -------
        ``BaseDataLoaderCreator``: The instantiated dataloader creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.experimental.dataloader import setup_dataloader_creator
        >>> creator = setup_dataloader_creator(
        ...     {
        ...         "_target_": "gravitorch.experimental.dataloader.VanillaDataLoaderCreator",
        ...         "dataset": {
        ...             "_target_": "gravitorch.datasets.DummyMultiClassDataset",
        ...             "num_examples": 10,
        ...             "num_classes": 2,
        ...             "feature_size": 4,
        ...         },
        ...     }
        ... )
        >>> creator
        VanillaDataLoaderCreator(
          dataset=DatasetCreator(
              cache=False
              dataset={'_target_': 'gravitorch.datasets.DummyMultiClassDataset', 'num_examples': 10, 'num_classes': 2, 'feature_size': 4}
            )
          seed=0
        )
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the dataloader creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataLoaderCreator.factory(**creator)
    return creator
