__all__ = ["setup_collator"]

from collections.abc import Callable
from typing import Union

from torch.utils.data.dataloader import default_collate

from gravitorch.data.dataloaders.collators.base import BaseCollator


def setup_collator(collator: Union[Callable, dict, None]) -> Callable:
    r"""Sets up a data loader collator.

    Args:
        collator (``Callable`` or dict or None): Specifies the
            data loader collator or its configuration. If ``None``,
            the default data loader collator is used.

    Returns:
        ``Callable``: The data loader collator.
    """
    if collator is None:
        collator = default_collate
    if isinstance(collator, dict):
        collator = BaseCollator.factory(**collator)
    return collator
