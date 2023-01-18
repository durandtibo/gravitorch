from __future__ import annotations

__all__ = ["BaseIterDataPipeCreator", "setup_iter_datapipe_creator"]

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseIterDataPipeCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement an ``IterDataPipe`` creator.

    A ``IterDataPipe`` creator is responsible to create a single
    DataPipe.

    Note: it is possible to create an ``IterDataPipe`` object without
    using this class.
    """

    @abstractmethod
    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe:
        r"""Creates an ``IterDataPipe`` object.

        Args:
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                ``IterDataPipe`` by using the current epoch value.
                Default: ``None``
            source_inputs (sequence or ``None``): Specifies the first
                positional arguments of the source ``IterDataPipe``.
                This argument can be used to create a new
                ``IterDataPipe`` object, that takes existing
                ``IterDataPipe`` objects as input. See examples below
                to see how to use it. If ``None``, ``source_inputs``
                is set to an empty tuple. Default: ``None``

        Returns:
            ``IterDataPipe``: The created ``IterDataPipe`` object.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.creators.datapipe import BaseIterDataPipeCreator
            >>> creator: BaseIterDataPipeCreator = ...  # Create an IterDataPipe creator
            # Create an IterDataPipe object
            >>> from torch.utils.data import IterDataPipe
            >>> my_datapipe: IterDataPipe = creator.create()
            # Create an IterDataPipe object with an engine
            >>> from gravitorch.engines import BaseEngine
            >>> my_engine: BaseEngine = ...  # Create an engine
            >>> my_datapipe: IterDataPipe = creator.create(my_engine)
            # Create an IterDataPipe object with some source_inputs
            >>> my_datapipe: IterDataPipe = creator.create(source_inputs=[...])
            # Create an IterDataPipe object with en engine and some source_inputs
            >>> my_datapipe: IterDataPipe = creator.create(engine=my_engine, source_inputs=[...])
        """


def setup_iter_datapipe_creator(creator: BaseIterDataPipeCreator | dict) -> BaseIterDataPipeCreator:
    r"""Sets up an ``IterDataPipe`` creator.

    The ``IterDataPipe`` creator is instantiated from its
    configuration by using the ``BaseDataPipeCreator`` factory
    function.

    Args:
        creator (``BaseIterDataPipeCreator`` or dict): Specifies the
            ``IterDataPipe`` creator or its configuration.

    Returns:
        ``BaseIterDataPipeCreator``: The instantiated ``IterDataPipe``
            creator.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.creators.datapipe import setup_iter_datapipe_creator
        # Set up an ``IterDataPipe`` creator from an ``IterDataPipe`` creator i.e. do nothing ;)
        >>> from gravitorch.creators.datapipe import SequentialIterDataPipeCreator
        >>> creator = setup_iter_datapipe_creator(
        ...     SequentialIterDataPipeCreator(
        ...         config=[
        ...             {
        ...                 OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapperIterDataPipe",
        ...                 "data": [1, 2, 3, 4],
        ...             }
        ...         ],
        ...     )
        ... )
        # Set up an ``IterDataPipe`` creator from its config
        >>> from objectory import OBJECT_TARGET
        >>> creator = setup_iter_datapipe_creator(
        ...     {
        ...         OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
        ...         "config": [
        ...             {
        ...                 OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapperIterDataPipe",
        ...                 "data": [1, 2, 3, 4],
        ...             }
        ...         ],
        ...     }
        ... )
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing a IterDataPipe creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseIterDataPipeCreator.factory(**creator)
    return creator
