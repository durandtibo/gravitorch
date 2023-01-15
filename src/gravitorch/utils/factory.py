__all__ = ["setup_distribution"]

import logging
from typing import Union

from objectory import factory
from torch.distributions import Distribution

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_distribution(distribution: Union[Distribution, dict]) -> Distribution:
    r"""Sets up a distribution.

    Args:
        distribution (``torch.distributions.Distribution`` or dict
            or None): Specifies the distribution or its configuration.

    Returns:
        ``torch.distributions.Distribution``: The distribution.
    """
    if isinstance(distribution, dict):
        logger.info(
            "Initializing a distribution from its configuration... "
            f"{str_target_object(distribution)}"
        )
        distribution = factory(**distribution)
    return distribution
