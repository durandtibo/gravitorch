r"""This file implements a module manager."""

from __future__ import annotations

__all__ = ["ModuleManager"]

import logging
from typing import Any

from gravitorch.utils.format import str_indent, str_torch_mapping

logger = logging.getLogger(__name__)


class ModuleManager:
    r"""Implements a module manager to easily manage a group of modules.

    This module manager assumes that the modules have a ``state_dict``
    and ``load_state_dict`` methods.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.module_manager import ModuleManager
        >>> manager = ModuleManager()
        >>> from torch import nn
        >>> manager.add_module("my_module", nn.Linear(4, 6))
        >>> manager.get_module("my_module")
        Linear(in_features=4, out_features=6, bias=True)
    """

    def __init__(self) -> None:
        self._modules = {}

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        if self._modules:
            return (
                f"{self.__class__.__qualname__}(\n"
                f"  {str_indent(str_torch_mapping(self._modules))}\n)"
            )
        return f"{self.__class__.__qualname__}()"

    def add_module(self, name: str, module: Any) -> None:
        r"""Adds a module to the module state manager.

        Note that the name should be unique. If the name exists, the
        old module will be overwritten by the new module.

        Args:
        ----
            name (str): Specifies the name of the module to add.
            module: Specifies the module to add.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module("my_module", nn.Linear(4, 6))
        """
        if name in self._modules:
            logger.info(f"Overriding the '{name}' module")
        self._modules[name] = module

    def get_module(self, name: str) -> Any:
        r"""Gets a module.

        Args:
        ----
            name (str): Specifies the module to get.

        Returns:
        -------
            The module

        Raises:
        ------
            ValueError if the module does not exist.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module("my_module", nn.Linear(4, 6))
            >>> manager.get_module("my_module")
            Linear(in_features=4, out_features=6, bias=True)
        """
        if not self.has_module(name):
            raise ValueError(f"The module '{name}' does not exist")
        return self._modules[name]

    def has_module(self, name: str) -> bool:
        r"""Indicates if there is module for the given name.

        Args:
        ----
            name (str): Specifies the name to check.

        Returns:
        -------
            bool: ``True`` if the module exists, otherwise ``False``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module("my_module", nn.Linear(4, 6))
            >>> manager.has_module("my_module")
            True
            >>> manager.has_module("missing")
            False
        """
        return name in self._modules

    def remove_module(self, name: str) -> None:
        r"""Removes a module.

        Args:
        ----
            name (str): Specifies the name of the module to remove.

        Raises:
        ------
            ValueError if the module does not exist.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module("my_module", nn.Linear(4, 6))
            >>> manager.remove_module("my_module")
            >>> manager.has_module("my_module")
            False
        """
        if name not in self._modules:
            raise ValueError(
                f"The module '{name}' does not exist so it is not possible to remove it"
            )
        self._modules.pop(name)

    def load_state_dict(self, state_dict: dict, keys: list | tuple | None = None) -> None:
        r"""Loads the state dict of each module.

        Note this method ignore the missing modules or keys. For
        example if you want to load the optimizer module but there is
        no 'optimizer' key in the state dict, this method will ignore
        the optimizer module.

        Args:
        ----
            state_dict (dict): Specifies the state dict to load.
            keys (list or tuple or ``None``): Specifies the keys to
                load. If ``None``, it loads all the keys associated
                to the registered modules.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> import torch
            >>> from torch import nn
            >>> manager.add_module("my_module", nn.Linear(4, 6))
            >>> manager.load_state_dict(
            ...     {"my_module": {"weight": torch.ones(6, 4), "bias": torch.zeros(6)}}
            ... )
        """
        keys = keys or tuple(self._modules.keys())
        for key in keys:
            if key not in state_dict:
                logger.info(f"Ignore key {key} because it is not in the state dict")
                continue
            if key not in self._modules:
                logger.info(f"Ignore key {key} because there is no module associated to it")
                continue
            if not hasattr(self._modules[key], "load_state_dict"):
                logger.info(
                    f"Ignore key {key} because the module does not have 'load_state_dict' method"
                )
                continue
            self._modules[key].load_state_dict(state_dict[key])

    def state_dict(self) -> dict:
        r"""Creates a state dict with all the modules.

        The state of each module is store with the associated key of
        the module.

        Returns
        -------
            dict: The state dict of all the modules.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module("my_module", nn.Linear(4, 6))
            >>> manager.state_dict()  # doctest: +ELLIPSIS
            {'my_module': OrderedDict([('weight', tensor([[...)),
                ('bias', tensor([...))])}
            >>> manager.add_module("int", 123)
            >>> manager.state_dict()  # doctest: +ELLIPSIS
            {'my_module': OrderedDict([('weight', tensor([[...]])),
                ('bias', tensor([...]))])}
        """
        state = {}
        for name, module in self._modules.items():
            if hasattr(module, "state_dict"):
                state[name] = module.state_dict()
            else:
                logger.info(f"Skip '{name}' module because it does not have 'state_dict' method")
        return state
