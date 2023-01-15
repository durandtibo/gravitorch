__all__ = ["setup_device_placement"]

import logging
from typing import Union

from gravitorch.utils.device_placement.base import BaseDevicePlacement
from gravitorch.utils.device_placement.noop import NoOpDevicePlacement
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_device_placement(
    device_placement: Union[BaseDevicePlacement, dict, None]
) -> BaseDevicePlacement:
    r"""Sets up a device placement module.

    The device placement module is instantiated from its configuration
    by using the ``BaseDevicePlacement`` factory function.

    Args:
        device_placement (``BaseDevicePlacement`` or dict or ``None``):
            Specifies the device placement module or its configuration.
            If ``None``, the ``NoOpDevicePlacement`` is instantiated.

    Returns:
        ``BaseDevicePlacement``: The device placement module.
    """
    if device_placement is None:
        device_placement = NoOpDevicePlacement()
    if isinstance(device_placement, dict):
        logger.info(
            "Initializing a device placement module from its configuration... "
            f"{str_target_object(device_placement)}"
        )
        device_placement = BaseDevicePlacement.factory(**device_placement)
    return device_placement
