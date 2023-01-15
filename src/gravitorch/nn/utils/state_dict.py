__all__ = [
    "find_module_state_dict",
    "load_checkpoint_to_module",
    "load_model_state_dict",
    "load_state_dict_to_module",
    "show_state_dict_info",
    "state_dicts_are_equal",
]

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import torch
from coola import objects_are_equal
from tabulate import tabulate
from torch.nn import Module

from gravitorch import constants as ct
from gravitorch.nn.utils.module_helpers import get_module_device
from gravitorch.utils.mapping import remove_keys_starting_with
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)


def find_module_state_dict(state_dict: Union[dict, list, tuple], module_keys: set) -> dict:
    r"""Tries to find automatically the part of the state dict related to a
    module.

    The user should specify the set of module's keys:
    ``set(module.state_dict().keys())``. This function assumes that
    the set of keys only exists at one location in the state dict.
    If the set of keys exists at several locations in the state dict,
    only the first one is returned.

    Args:
        state_dict (dict, list or tuple): Specifies the state dict.
            This function is called recursively on this input to find
            the queried state dict.
        module_keys (set): Specifies the set of module keys.

    Returns:
        dict: The part of the state dict related to a module if it is
            found, otherwise an empty dict.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import find_module_state_dict
        >>> state = {
        ...     'model': {
        ...         'network': {
        ...             'weight': torch.ones(5, 4),
        ...             'bias': 2 * torch.ones(5),
        ...         }
        ...     }
        ... }
        >>> module = torch.nn.Linear(4, 5)
        >>> find_module_state_dict(state, module_keys=set(module.state_dict().keys()))
        {'weight': torch.ones(5, 4), 'bias': 2 * torch.ones(5)}
    """
    if isinstance(state_dict, dict):
        if set(state_dict.keys()) == module_keys:
            return state_dict
        for value in state_dict.values():
            state_dict = find_module_state_dict(value, module_keys)
            if state_dict:
                return state_dict
    elif isinstance(state_dict, (list, tuple, set)):
        for value in state_dict:
            state_dict = find_module_state_dict(value, module_keys)
            if state_dict:
                return state_dict
    return {}


def load_checkpoint_to_module(
    path: Union[Path, str],
    module: Module,
    strict: bool = True,
    key: Optional[Union[str, Sequence[str]]] = None,
) -> None:
    r"""Loads the weights store in a checkpoint file into the given module.

    The checkpoint should contain the state dict of the module, which
    is the recommended way to store the model state/weights
    https://pytorch.org/docs/stable/notes/serialization.html#serialization-semantics.
    This function raises an error (``RuntimeError``) if the weights
    are not compatible with the module. This function assumes the
    model is on a single device and may not work in a multi-device
    setting. You will have to create your own function to manage the
    multi-device setting. This function automatically moves the
    weights to the module device if they are on another device.

    Args:
        module (``torch.nn.Module``): Specifies the module. This
            function changes the weights of this module.
        path (str): Specifies the path to the checkpoint with the
            module weights. The checkpoint should be a PyTorch file.
        strict (bool, optional): whether to strictly enforce that the
            keys in ``state_dict`` match the keys returned by this
            module's :meth:`~torch.nn.Module.state_dict` function.
            Default: ``True``
        key (str or list or tuple or ``None``, optional): Specifies
            the key of the state dict to load. The state dict can
            contain data that are not about the module, so they may
            need to be excluded. For nested case, it is possible to
            specify the list of keys to get the good part of the state
            dict. Default: ``None``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import load_checkpoint_to_module
        >>> net = torch.nn.Linear(4, 5)
        >>> load_checkpoint_to_module(path="/path/to/checkpoint.pt", module=net)
        # To load only the value defined at state_dict["model"]
        >>> load_checkpoint_to_module(path="/path/to/checkpoint.pt", module=net, key="model")
        # To load only the value defined at state_dict["model"]["network"]
        >>> load_checkpoint_to_module(
        ...     path="/path/to/checkpoint.pt",
        ...     module=net,
        ...     key=["model", "network"],
        ... )
        >>> load_checkpoint_to_module(
        ...     path="/path/to/checkpoint.pt",
        ...     module=net,
        ...     key=("model", "network"),
        ... )
    """
    # Get the device of the module. It assumes the module is on a single device.
    device = get_module_device(module)
    # Load the state dict.
    path = sanitize_path(path)
    logger.info(f"Loading checkpoint from {path}...")
    state_dict = torch.load(path, map_location=device)

    if key:  # Go to a given key of the state dict
        key = (key,) if isinstance(key, str) else key
        for k in key:
            state_dict = state_dict[k]

    load_state_dict_to_module(state_dict=state_dict, module=module, strict=strict)


def load_state_dict_to_module(state_dict: dict, module: Module, strict: bool = True) -> None:
    r"""Loads a state dict into a given module.

    This function will automatically try to find the module state dict
    in the given state dict.

    Args:
        state_dict (dict): Specifies the state dict with the module
            state dict.
        module (``torch.nn.Module``): Specifies the module. This
            function changes the weights of this module.
        strict (bool, optional): whether to strictly enforce that the
            keys in ``state_dict`` match the keys returned by this
            module's :meth:`~torch.nn.Module.state_dict` function.
            Default: ``True``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.utils import load_state_dict_to_module
        >>> module = torch.nn.Linear(4, 5)
        >>> state_dict = {
        ...     "weight": torch.tensor(
        ...         [
        ...             [-0.0328, 0.1781, -0.1038, 0.4252],
        ...             [0.0242, -0.2810, 0.3932, 0.2912],
        ...             [-0.0474, -0.0365, -0.0279, -0.1859],
        ...             [0.2816, 0.2591, -0.3863, 0.4778],
        ...             [0.3242, 0.0191, 0.0883, 0.0032],
        ...         ]
        ...     ),
        ...     "bias": torch.tensor([0.4347, -0.3001, -0.3297, 0.2118, -0.3951]),
        ... }
        >>> load_state_dict_to_module(path="/path/to/checkpoint.pt", module=module)
    """
    try:
        module.load_state_dict(state_dict, strict)
    except RuntimeError:
        logger.warning(
            "Could not load the state dict. Try to find automatically the part of the state dict "
            "that matches with the module."
        )
        state_dict = find_module_state_dict(state_dict, set(module.state_dict().keys()))
        if not state_dict:
            logger.info("Could not find a part of the state dict that matches with the module.")
            raise
        logger.info(
            "Found a part of the state dict that matches with the module. Try to load it in the "
            "module."
        )
        module.load_state_dict(state_dict, strict)
    logger.info("The weights are loaded in the module.")


def state_dicts_are_equal(module1: Module, module2: Module) -> bool:
    r"""Indicates if the state dict of 2 modules are equal or not.

    Args:
        module1 (``torch.nn.Module``): Specifies the first module to
            compare.
        module2 (``torch.nn.Module``): Specifies the second module to
            compare.

    Returns:
        bool: ``True`` if the state dict of 2 modules are equal,
            otherwise ``False``
    """
    return objects_are_equal(module1.state_dict(), module2.state_dict())


def show_state_dict_info(state_dict: dict, tablefmt: str = "simple") -> None:
    r"""Shows information about the state dict.

    Args:
        state_dict (dict): Specifies the state dict to analyze.
        tablefmt (str, optional): Specifies the table format.
            Default: ``'simple'``
    """
    stats = [["key", "shape", "dtype"]]
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            stats.append([key, list(value.shape), value.dtype])
    logger.info(f'State dict info\n{tabulate(stats, headers="firstrow", tablefmt=tablefmt)}\n')


def load_model_state_dict(
    path: Union[Path, str],
    module: Module,
    exclude_key_prefixes: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> None:
    r"""Loads a model state dict.

    Args:
        path (``pathlib.Path`` or str): Specifies the path to the
            checkpoint with the module weights. The checkpoint
            should be a PyTorch file.
        module (``torch.nn.Module``): Specifies the module. This
            function changes the weights of this module.
        exclude_key_prefixes (sequence or ``None``, optional):
            Specifies the list of key prefixes to exclude when
            loading the state dict. Default: ``None``
        strict (bool, optional): whether to strictly enforce that
            the keys in ``state_dict`` match the keys returned by
            this module's :meth:`~torch.nn.Module.state_dict`
            function. Default: ``True``
    """
    # Load the state dict.
    path = sanitize_path(path)
    logger.info(f"Loading checkpoint from {path}...")
    # Get the device of the module. It assumes the module is on a single device.
    device = get_module_device(module)
    state_dict = torch.load(path, map_location=device)

    # Prepare the state dict
    state_dict = state_dict["modules"][ct.MODEL]  # TODO: make more generic
    exclude_key_prefixes = exclude_key_prefixes or []
    for prefix in exclude_key_prefixes:
        state_dict = remove_keys_starting_with(state_dict, prefix)

    show_state_dict_info(state_dict)
    module.load_state_dict(state_dict, strict)
