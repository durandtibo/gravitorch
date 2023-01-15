__all__ = ["BaseDevicePlacement"]

from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory


class BaseDevicePlacement(ABC, metaclass=AbstractFactory):
    r"""Defines a base class to send object on a target device.

    Note that the object should be sent to a single target device. This
    class cannot be used if you want to send an object to multiple
    target devices.
    """

    @abstractmethod
    def send(self, obj: Any) -> Any:
        r"""Sends the object on a target device.

        Args:
            obj: The object to send to the target device.

        Returns:
            The object on the target device.
        """
