__all__ = [
    "CpuDevicePlacement",
    "CudaDevicePlacement",
    "ManualDevicePlacement",
    "MpsDevicePlacement",
]

from typing import Any, Union

import torch

from gravitorch.utils import move_to_device
from gravitorch.utils.device_placement.base import BaseDevicePlacement


class ManualDevicePlacement(BaseDevicePlacement):
    r"""Implements a device placement class to send objects on a given device.

    The user is responsible to choose the target device.
    """

    def __init__(self, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(device={self.device})"

    def send(self, obj: Any) -> Any:
        r"""Sends the object on a target device.

        Args:
            obj: The object to send to the target device.

        Returns:
            The object on the target device.
        """
        return move_to_device(obj, self.device)


class CpuDevicePlacement(ManualDevicePlacement):
    r"""Implements a device placement class to send objects on a cpu device."""

    def __init__(self):
        super().__init__(torch.device("cpu"))


class CudaDevicePlacement(ManualDevicePlacement):
    r"""Implements a device placement class to send objects on a cuda device.

    Args:
        index (int, optional): Specifies the index of the cuda device.
            Default: ``0``
    """

    def __init__(self, index: int = 0):
        super().__init__(torch.device(type="cuda", index=index))


class MpsDevicePlacement(ManualDevicePlacement):
    r"""Implements a device placement class to send objects on a Metal
    Performance Shaders (MPS) device."""

    def __init__(self):
        super().__init__(torch.device("mps"))
