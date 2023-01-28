__all__ = ["BaseResourceManager", "setup_resource_manager"]

import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import Union

from objectory import AbstractFactory

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseResourceManager(AbstractContextManager, metaclass=AbstractFactory):
    r"""Defines the base class to manage a resource."""

    @abstractmethod
    def configure(self) -> None:
        r"""Configure the resource."""

    @abstractmethod
    def show(self) -> None:
        r"""Shows the information associated to resource."""


def setup_resource_manager(
    resource_manager: Union[BaseResourceManager, dict]
) -> BaseResourceManager:
    r"""Sets up the resource manager.

    The resource manager is instantiated from its configuration by using the
    ``BaseResourceManager`` factory function.

    Args:
        resource_manager (``BaseResourceManager`` or dict): Specifies
            the resource manager or its configuration.

    Returns:
        ``BaseResourceManager``: The instantiated resource manager.
    """
    if isinstance(resource_manager, dict):
        logger.debug(
            "Initializing a resource manager from its configuration... "
            f"{str_target_object(resource_manager)}"
        )
        resource_manager = BaseResourceManager.factory(**resource_manager)
    return resource_manager
