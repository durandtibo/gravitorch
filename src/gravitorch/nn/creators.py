from __future__ import annotations

__all__ = ["create_sequential"]

from collections.abc import Sequence

from torch.nn import Module, Sequential

from gravitorch.nn.utils.factory import setup_module


def create_sequential(modules: Sequence[Module | dict]) -> Sequential:
    r"""Creates a ``torch.nn.Sequential`` from a sequence of modules.

    Args:
    ----
        modules (sequence): Specifies the sequence of modules or their
            configuration.

    Returns:
    -------
        ``torch.nn.Sequential``: The instantiated
            ``torch.nn.Sequential`` module.
    """
    return Sequential(*[setup_module(module) for module in modules])
