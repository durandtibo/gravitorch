from __future__ import annotations

__all__ = ["BaseDataCreator", "is_datacreator_config", "setup_datacreator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataCreator(ABC, Generic[T], metaclass=AbstractFactory):
    r"""Defines the base class to implement a data creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.datacreators import HypercubeVertexDataCreator
        >>> creator = HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=6)
        >>> creator
        HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=6, noise_std=0.2, random_seed=15782179921860610490)
        >>> data = creator.create()
        >>> data
        {'target': tensor([...]), 'input': tensor([[...]])}
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> T:
        r"""Creates data.

        Args:
        ----
            engine (``BaseEngine`` or ``None``): Specifies an engine.
                Default: ``None``

        Returns:
        -------
            The created data.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.data.datacreators import HypercubeVertexDataCreator
            >>> creator = HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=6)
            >>> data = creator.create()
            >>> data
            {'target': tensor([...]), 'input': tensor([[...]])}
        """


def is_datacreator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseDataCreator``.

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
            for a ``BaseDataCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.datacreators import is_datacreator_config
        >>> is_datacreator_config(
        ...     {"_target_": "gravitorch.data.datacreators.HypercubeVertexDataCreator"}
        ... )
        True
    """
    return is_object_config(config, BaseDataCreator)


def setup_datacreator(data_creator: BaseDataCreator | dict) -> BaseDataCreator:
    r"""Sets up a data creator.

    The data creator is instantiated from its configuration by using
    the ``BaseDataCreator`` factory function.

    Args:
    ----
        data_creator (``BaseDataCreator`` or dict): Specifies the data
            creator or its configuration.

    Returns:
    -------
        ``BaseDataCreator``: The instantiated data creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.datacreators import setup_datacreator
        >>> datacreator = setup_datacreator(
        ...     {
        ...         "_target_": "gravitorch.data.datacreators.HypercubeVertexDataCreator",
        ...         "num_examples": 10,
        ...         "num_classes": 5,
        ...         "feature_size": 6,
        ...     }
        ... )
        >>> datacreator
        HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=6, noise_std=0.2, random_seed=15782179921860610490)
    """
    if isinstance(data_creator, dict):
        logger.info(
            "Initializing a data creator from its configuration... "
            f"{str_target_object(data_creator)}"
        )
        data_creator = BaseDataCreator.factory(**data_creator)
    return data_creator
