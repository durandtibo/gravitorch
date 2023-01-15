__all__ = ["NoOpDevicePlacement"]

from typing import Any

from gravitorch.utils.device_placement.base import BaseDevicePlacement


class NoOpDevicePlacement(BaseDevicePlacement):
    r"""Implements a no-operation device placement.

    This class does not change the device placement of the object or in
    other words, the target device is always the device where the object
    is.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def send(self, obj: Any) -> Any:
        r"""Sends the object on a target device.

        Args:
            obj: The object to send to the target device.

        Returns:
            The object on the target device.
        """
        return obj
