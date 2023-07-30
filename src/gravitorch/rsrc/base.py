from __future__ import annotations

__all__ = ["BaseResource", "setup_resource"]

import logging
from contextlib import AbstractContextManager

from objectory import AbstractFactory

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseResource(AbstractContextManager, metaclass=AbstractFactory):
    r"""Defines the base class to manage a resource."""


def setup_resource(resource: BaseResource | dict) -> BaseResource:
    r"""Sets up a resource.

    The resource manager is instantiated from its configuration by using the
    ``BaseResource`` factory function.

    Args:
    ----
        resource (``BaseResource`` or dict): Specifies
            the resource or its configuration.

    Returns:
    -------
        ``BaseResource``: The instantiated resource.
    """
    if isinstance(resource, dict):
        logger.debug(
            f"Initializing a resource from its configuration... {str_target_object(resource)}"
        )
        resource = BaseResource.factory(**resource)
    if not isinstance(resource, BaseResource):
        logger.warning(f"resource is not a BaseResource (received: {type(resource)})")
    return resource
