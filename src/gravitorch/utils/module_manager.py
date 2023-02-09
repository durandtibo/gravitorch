r"""This file implements a module manager."""

__all__ = ["ModuleManager"]

import logging
from typing import Any, Union

from gravitorch.utils.format import str_indent, to_torch_mapping_str

logger = logging.getLogger(__name__)


class ModuleManager:
    r"""Implements a module manager to easily manage a group of modules.

    This module manager assumes that the modules have a ``state_dict``
    and ``load_state_dict`` methods.
    """

    def __init__(self):
        self._modules = {}

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        if self._modules:
            return (
                f"{self.__class__.__qualname__}(\n"
                f"  {str_indent(to_torch_mapping_str(self._modules))}\n)"
            )
        return f"{self.__class__.__qualname__}()"

    def add_module(self, name: str, module: Any) -> None:
        r"""Adds a module to the module state manager.

        Note that the name should be unique. If the name exists, the
        old module will be overwritten by the new module.

        Args:
            name (str): Specifies the name of the module to add.
            module: Specifies the module to add.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module('my_module', nn.Linear(4, 6))
        """
        if name in self._modules:
            logger.info(f"Overriding the '{name}' module")
        self._modules[name] = module

    def get_module(self, name: str) -> Any:
        r"""Gets a module.

        Args:
            name (str): Specifies the module to get.

        Returns:
            The module

        Raises:
            ValueError if the module does not exist.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module('my_module', nn.Linear(4, 6))
            >>> manager.get_module('my_module')
            nn.Linear(4, 6)
        """
        if not self.has_module(name):
            raise ValueError(f"The module {name} does not exist")
        return self._modules[name]

    def has_module(self, name: str) -> bool:
        r"""Indicates if there is module for the given name.

        Args:
            name (str): Specifies the name to check.

        Returns:
            bool: ``True`` if the module exists, otherwise ``False``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module('my_module', nn.Linear(4, 6))
            >>> manager.has_module('my_module')
            True
            >>> manager.has_module('missing')
            False
        """
        return name in self._modules

    def remove_module(self, name: str) -> None:
        r"""Removes a module.

        Args:
            name (str): Specifies the name of the module to remove.

        Raises:
            ValueError if the module does not exist.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module('my_module', nn.Linear(4, 6))
            >>> manager.remove_module('my_module')
            >>> manager.has_module('my_module')
            False
        """
        if name not in self._modules:
            raise ValueError(f"The module {name} does not exist so it is not possible to remove it")
        self._modules.pop(name)

    def load_state_dict(self, state_dict: dict, keys: Union[list, tuple, None] = None) -> None:
        r"""Loads the state dict of each module.

        Note this method ignore the missing modules or keys. For
        example if you want to load the optimizer module but there is
        no 'optimizer' key in the state dict, this method will ignore
        the optimizer module.

        Args:
            state_dict (dict): Specifies the state dict to load.
            keys (list or tuple or ``None``): Specifies the keys to
                load. If ``None``, it loads all the keys associated
                to the registered modules.
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

        Returns:
            dict: The state dict of all the modules.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.module_manager import ModuleManager
            >>> manager = ModuleManager()
            >>> from torch import nn
            >>> manager.add_module('my_module', nn.Linear(4, 6))
            >>> manager.state_dict()
            {'my_module': OrderedDict([('weight', tensor([[-0.3810, -0.3761,  0.3272,  0.2582],
                    [ 0.4013,  0.0173,  0.3931, -0.1642],
                    [-0.4772,  0.1962, -0.0101,  0.0654],
                    [-0.1067,  0.3788, -0.0949, -0.3952],
                    [ 0.0111, -0.2536, -0.3626, -0.0810],
                    [-0.1757, -0.4256, -0.1076, -0.2050]])),
                ('bias', tensor([ 0.2897, -0.4282, -0.2547, -0.2704, -0.1545,  0.2825]))])}
            >>> manager.add_module('int', 123)
            >>> manager.state_dict()
            {'my_module': OrderedDict([('weight', tensor([[-0.3810, -0.3761,  0.3272,  0.2582],
                    [ 0.4013,  0.0173,  0.3931, -0.1642],
                    [-0.4772,  0.1962, -0.0101,  0.0654],
                    [-0.1067,  0.3788, -0.0949, -0.3952],
                    [ 0.0111, -0.2536, -0.3626, -0.0810],
                    [-0.1757, -0.4256, -0.1076, -0.2050]])),
                ('bias', tensor([ 0.2897, -0.4282, -0.2547, -0.2704, -0.1545,  0.2825]))])}
        """
        state = {}
        for name, module in self._modules.items():
            if hasattr(module, "state_dict"):
                state[name] = module.state_dict()
            else:
                logger.info(f"Skip '{name}' module because it does not have 'state_dict' method")
        return state
