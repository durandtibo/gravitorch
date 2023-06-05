__all__ = ["create_compose"]

from collections.abc import Callable, Sequence
from typing import Union

from gravitorch.utils import setup_object
from gravitorch.utils.imports import check_torchvision, is_torchvision_available

if is_torchvision_available():
    from torchvision.transforms import Compose
else:
    Compose = None


def create_compose(transforms: Sequence[Union[Callable, dict]]) -> Compose:
    r"""Instantiates a ``torchvision.transforms.Compose`` from its configuration.

    Args:
        transforms (sequence of ``Transform`` objects): Specifies the
            sequence of transforms (or their configuration) to compose.

    Returns:
        ``torchvision.transforms.Compose``: The instantiated
            composition of transforms

    Raises:
        RuntimeError if ``torchvision`` is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.transforms.vision import create_compose
        >>> from torchvision.transforms import PILToTensor
        >>> create_compose(
        ...     [
        ...         {'_target_': "torchvision.transforms.CenterCrop", "size": 10},
        ...         PILToTensor(),
        ...     ]
        ... )
    """
    check_torchvision()
    return Compose([setup_object(transform) for transform in transforms])
