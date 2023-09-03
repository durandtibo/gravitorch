from __future__ import annotations

__all__ = ["BaseDataCreator", "setup_data_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory

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
        >>> data  # doctest: +ELLIPSIS
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
            >>> data  # doctest: +ELLIPSIS
            {'target': tensor([...]), 'input': tensor([[...]])}
        """


def setup_data_creator(data_creator: BaseDataCreator | dict) -> BaseDataCreator:
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

        >>> from gravitorch.data.datacreators import setup_data_creator
        >>> datacreator = setup_data_creator(
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
