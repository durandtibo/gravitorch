r"""This module contains some utility functions around ``torch.device``."""

__all__ = ["get_available_devices", "move_to_device"]

from typing import TypeVar

import torch
from torch.nn.utils.rnn import PackedSequence

T = TypeVar("T")


def get_available_devices() -> tuple[str, ...]:
    r"""Gets the available PyTorch devices on the machine.

    Returns:
        tuple: The available devices.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.torch_device import get_available_devices
        >>> get_available_devices()
        ('cpu', 'cuda:0')
    """
    if torch.cuda.is_available():
        return ("cpu", "cuda:0")
    return ("cpu",)


def move_to_device(data: T, device: torch.device) -> T:
    r"""Moves an object to a given device.

    If the object is a nested object (e.g. list, tuple, dictionary,
    set), this function sends the elements to the device. The current
    implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``set``
        - ``torch.Tensor``
        - ``torch.nn.Module``
        - ``torch.nn.utils.rnn.PackedSequence``
        - ``tuple``

    Based on https://github.com/huggingface/accelerate

    Args:
        data: Specifies the data to move to the device. If it is a
            nested object, the data is moved recursively to the
            device.
        device (``torch.device``): Specifies the device to send the
            data to.

    Returns:
        The object on the given device. If it is not possible to move
            the object to the device, the input object is returned.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.torch_device import move_to_device
        >>> move_to_device(
        ...     {'tensor1': torch.ones(2, 3), 'tensor2': torch.zeros(4)},
        ...     device=torch.device('cuda:0'),
        ... )
        {'tensor1': tensor([[1., 1., 1.], [1., 1., 1.]], device='cuda:0'),
         'tensor2': tensor([0., 0., 0., 0.], device='cuda:0')}
    """
    if isinstance(data, PackedSequence):
        return data.to(device)
    if isinstance(data, (list, tuple, set)):
        return type(data)(move_to_device(t, device) for t in data)
    if isinstance(data, dict):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    if not hasattr(data, "to"):
        return data
    return data.to(device)
