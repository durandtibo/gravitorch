__all__ = [
    "freeze_module",
    "get_module_device",
    "get_module_devices",
    "get_module_input_size",
    "get_module_name",
    "get_module_output_size",
    "has_batch_norm",
    "has_learnable_parameters",
    "has_parameters",
    "is_batch_first",
    "is_module_on_device",
    "module_mode",
    "num_learnable_parameters",
    "num_parameters",
    "top_module_mode",
    "unfreeze_module",
]

from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import nn


def has_parameters(module: nn.Module) -> bool:
    r"""Indicates if the module has parameters.

    Args:
        module (``torch.nn.Module``): Specifies the module to test.

    Returns:
        bool: ``True`` if the module has at least one parameter,
            ``False`` otherwise.
    """
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def has_learnable_parameters(module: nn.Module) -> bool:
    r"""Indicates if the module has learnable parameters.

    Args:
        module (``torch.nn.Module``): Specifies the module to test.

    Returns:
        bool: ``True`` if the module has at least one learnable
            parameter, ``False`` otherwise.
    """
    return num_learnable_parameters(module) > 0


def num_parameters(module: nn.Module) -> int:
    r"""Computes the number of parameters.

    Args:
        module (``torch.nn.Module``): Specifies the module to compute
            the number of parameters.

    Returns:
        int: The number of parameters.
    """
    return sum(params.numel() for params in module.parameters())


def num_learnable_parameters(module: nn.Module) -> int:
    r"""Computes the number of learnable parameters.

    Args:
        module (``torch.nn.Module``): Specifies the module to compute
            the number of learnable parameters..

    Returns:
        int: The number of learnable parameters.
    """
    return sum(params.numel() for params in module.parameters() if params.requires_grad)


def freeze_module(module: nn.Module) -> None:
    r"""Freezes the parameters of the given module.

    Args:
        module (``torch.nn.Module``): Specifies the module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    r"""Unfreezes the parameters of the given module.

    Args:
        module (``torch.nn.Module``): Specifies the module to
            unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True


def get_module_device(module: nn.Module) -> torch.device:
    r"""Get the device used by this module.

    This function assumes the module uses a single device. If the
    module uses several devices, you should use
    ``get_module_devices``. It returns ``torch.device('cpu')`` if
    the model does not have parameters.

    Args:
        module (``torch.nn.Module``): Specifies the module.

    Returns:
        ``torch.device``: The device
    """
    if not has_parameters(module):
        return torch.device("cpu")
    return next(module.parameters()).device


def get_module_devices(module: nn.Module) -> tuple[torch.device, ...]:
    r"""Get the devices used in a module.

    Args:
        module (``torch.nn.Module``): Specifies the module.

    Returns:
        tuple: The tuple of ``torch.device``s used in the module.
    """
    devices = set()
    for param in module.parameters():
        devices.add(param.device)
    return tuple(devices)


def is_module_on_device(module: nn.Module, device: torch.device) -> bool:
    r"""Indicates if all the parameters of a module are on the specified device.

    Args:
        module (``torch.nn.Module``): Specifies the module.
        device (``torch.device``): Specifies the device.

    Returns:
        bool: ``True`` if all the parameters of the module are on the
            specified device, otherwise ``False``.
    """
    return all([p.device == device for p in module.parameters()])


def get_module_input_size(module: nn.Module) -> int:
    r"""Gets the input size of a module.

    This function works only for the module with a single input size.

    Note: it is an experimental function that supports a limited
    number of modules:

        - ``ModuleList``: assumes all the modules have the same
            input size.

    Args:
        module (``torch.nn.Module``): Specifies the input module.

    Returns:
        int: The input size of the module.

    Raises:
        TypeError: if the module is not supported.
    """
    if hasattr(module, "input_size"):
        return module.input_size
    if hasattr(module, "in_features"):
        return module.in_features
    if hasattr(module, "in_channels"):
        return module.in_channels
    if isinstance(module, nn.Sequential):
        return _get_sequential_input_size(module)
    if isinstance(module, nn.ModuleList) and len(module) > 0:
        return get_module_input_size(module[0])
    if isinstance(module, nn.MultiheadAttention):
        return module.embed_dim
    if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        return get_module_input_size(module.self_attn)
    if isinstance(module, (nn.TransformerEncoder, nn.TransformerDecoder)):
        return get_module_input_size(module.layers)

    raise TypeError(f"{type(module)} module is not supported")


def _get_sequential_input_size(sequential: nn.Sequential) -> int:
    r"""Gets the input size of a ``torch.nn.Sequential``.

    This function finds the first layer of the ``torch.nn.Sequential``
    for which it is possible to compute the input size.

    Args:
        sequential (``torch.nn.Sequential``): Specifies the input
            module.

    Returns:
        int: The input size of the ``torch.nn.Sequential`` object
            if possible.

    Raises:
        TypeError if the input is not a ``torch.nn.Sequential`` or
            if none of the child modules are supported.
    """
    if not isinstance(sequential, nn.Sequential):
        raise TypeError(
            f"Only the `torch.nn.Sequential` inputs are supported. (received: {sequential})"
        )
    for module in sequential:
        try:
            return get_module_input_size(module)
        except TypeError:
            pass
    raise TypeError(
        "It is not possible to find the input size because none of the child modules are supported"
    )


def get_module_output_size(module: nn.Module) -> int:
    r"""Gets the output size of a module.

    This function works only for the module with a single output size.

    Note: it is an experimental function that supports a limited
    number of modules:

        - ``ModuleList``: assumes all the modules have the same
            output  size.

    Args:
        module (``torch.nn.Module``): Specifies the input module.

    Returns:
        int: The output size of the module if possible.

    Raises:
        TypeError: if the module is not supported.
    """
    if hasattr(module, "output_size"):
        return module.output_size
    if hasattr(module, "out_features"):
        return module.out_features
    if hasattr(module, "out_channels"):
        return module.out_channels
    if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
        return module.hidden_size
    if isinstance(module, nn.Embedding):
        return module.embedding_dim
    if isinstance(module, nn.Sequential):
        return _get_sequential_output_size(module)
    if isinstance(module, nn.ModuleList) and len(module) > 0:
        return get_module_output_size(module[0])
    if isinstance(module, nn.MultiheadAttention):
        return module.embed_dim
    if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        return get_module_output_size(module.self_attn)
    if isinstance(module, (nn.TransformerEncoder, nn.TransformerDecoder)):
        return get_module_output_size(module.layers)

    raise TypeError(f"{type(module)} module is not supported")


def _get_sequential_output_size(sequential: nn.Sequential) -> int:
    r"""Gets the output size of a ``torch.nn.Sequential``.

    This function finds the last layer of the ``torch.nn.Sequential``
    for which it is possible to compute the output size.

    Args:
        sequential (``torch.nn.Sequential``): Specifies the input
            module.

    Returns:
        int: The output size of the ``torch.nn.Sequential`` object
            if possible.

    Raises:
        TypeError if the input is not a ``torch.nn.Sequential`` or
            if none of the child modules are supported.
    """
    if not isinstance(sequential, nn.Sequential):
        raise TypeError(
            f"Only the `torch.nn.Sequential` inputs are supported. (received: {sequential})"
        )
    for module in sequential[::-1]:
        try:
            return get_module_output_size(module)
        except TypeError:
            pass
    raise TypeError(
        "It is not possible to find the output size because none of the child modules are supported"
    )


def has_batch_norm(module: nn.Module) -> bool:
    r"""Indicates if the module has at least one batch norm layer.

    This function only verifies if the module has at least one of the
    following layer:

        - ``torch.nn.BatchNorm1d``
        - ``torch.nn.BatchNorm2d``
        - ``torch.nn.BatchNorm3d``
        - ``torch.nn.SyncBatchNorm``

    Args:
        module (``torch.nn.Module``): Specifies the input module.

    Returns:
        bool: ``True`` if the module has at least one batch norm layer,
            otherwise ``False``
    """
    for layer in module.modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            return True
    return False


def get_module_name(layer: nn.Module) -> str:
    r"""Gets the name of a module.

    The name of the module is the class name lower case.

    Args:
        layer (``torch.nn.Module``): Specifies the layer.

    Returns:
        str: The layer name.
    """
    return layer.__class__.__qualname__


def is_batch_first(module: nn.Module) -> bool:
    r"""Indicates if the input and output tensors are provided as ``(batch_size,
    seq_len, *)`` instead of ``(seq_len, batch_size,

    *)``.

    Args:
        module (``torch.nn.Module``): Specifies the module to check.

    Returns:
        bool: ``True`` if batch first, otherwise ``False``.

    Example usage:

    .. code-block:: python

        >>> from torch import nn
        >>> from gravitorch.nn import is_batch_first
        >>> is_batch_first(nn.MultiheadAttention(4, 1, batch_first=False))
        False
        >>> is_batch_first(nn.MultiheadAttention(4, 1, batch_first=True))
        True
    """
    if hasattr(module, "batch_first"):
        return module.batch_first
    if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        return is_batch_first(module.self_attn)
    if isinstance(module, (nn.TransformerEncoder, nn.TransformerDecoder)):
        return is_batch_first(module.layers[0])
    raise TypeError(f"{type(module)} module is not supported")


@contextmanager
def module_mode(module: nn.Module) -> Generator[None, None, None]:
    r"""Implements a context manager that restores the mode (train or eval) of
    every submodule individually.

    This context manager only restores the mode at the top-level.

    Args:
        module (``torch.nn.Module``): Specifies the module to restore
            the mode.

    Example usage:

    .. code-block:: python

        >>> from torch import nn
        >>> from gravitorch.nn import module_mode
        >>> module = nn.ModuleDict({'module1': nn.Linear(4, 6), 'module2': nn.Linear(2, 4).eval()})
        >>> print(module['module1'].training, module['module2'].training)
        True False
        >>> with module_mode(module):
        ...     module.eval()
        ...     print(module['module1'].training, module['module2'].training)
        False False
        >>> print(module['module1'].training, module['module2'].training)
        True False
    """
    modes = {}
    for name, submodule in module.named_modules():
        modes[name] = submodule.training
    try:
        yield
    finally:
        for name, submodule in module.named_modules():
            submodule.train(modes[name])


@contextmanager
def top_module_mode(module: nn.Module) -> Generator[None, None, None]:
    r"""Implements a context manager that restores the mode (train or eval) of a
    given module.

    This context manager only restores the mode at the top-level.

    Args:
        module (``torch.nn.Module``): Specifies the module to restore
            the mode.

    Example usage:

    .. code-block:: python

        >>> from torch import nn
        >>> from gravitorch.nn import module_mode
        >>> module = nn.Linear(4, 6)
        >>> print(module.training)
        True
        >>> with top_module_mode(module):
        ...     module.eval()
        ...     print(module)
        False
        >>> print(module.training)
        True
    """
    mode = module.training
    try:
        yield
    finally:
        module.train(mode)
