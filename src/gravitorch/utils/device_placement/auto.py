__all__ = ["AutoDevicePlacement"]

from gravitorch import distributed as dist
from gravitorch.utils.device_placement.manual import ManualDevicePlacement


class AutoDevicePlacement(ManualDevicePlacement):
    r"""Implements a device placement class that automatically find the "best"
    device to use.

    It uses a cuda device if cuda is available, otherwise it uses a cpu
    device.
    """

    def __init__(self):
        super().__init__(dist.device())
