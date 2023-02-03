__all__ = ["BaseResource", "setup_resource"]

import logging
from contextlib import AbstractContextManager
from typing import Union

from objectory import AbstractFactory

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseResource(AbstractContextManager, metaclass=AbstractFactory):
    r"""Defines the base class to manage a resource."""


def setup_resource(resource: Union[BaseResource, dict]) -> BaseResource:
    r"""Sets up a resource.

    The resource manager is instantiated from its configuration by using the
    ``BaseResource`` factory function.

    Args:
        resource (``BaseResource`` or dict): Specifies
            the resource or its configuration.

    Returns:
        ``BaseResource``: The instantiated resource.
    """
    if isinstance(resource, dict):
        logger.debug(
            f"Initializing a resource from its configuration... {str_target_object(resource)}"
        )
        resource = BaseResource.factory(**resource)
    return resource
