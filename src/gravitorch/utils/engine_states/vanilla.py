from __future__ import annotations

__all__ = ["EngineState"]

from typing import Any

from coola.utils import str_indent, str_mapping

from gravitorch.utils.asset import AssetManager
from gravitorch.utils.engine_states.base import BaseEngineState
from gravitorch.utils.history import BaseHistory, HistoryManager


class EngineState(BaseEngineState):
    r"""Defines the vanilla/default engine state.

    Args:
    ----
        epoch (int, optional): Specifies the number of epochs
            performed. Default: ``-1``.
        iteration (int, optional): Specifies the number of training
            iterations performed. Default: ``-1``.
        max_epochs (int, optional): Specifies the maximum number of
            epochs. Default: ``1``.
        random_seed (int, optional): Specifies the random seed.
            Default: ``9984043075503325450``.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.engine_states import EngineState
        >>> state = EngineState()
        >>> state
        EngineState(
          (modules): AssetManager(num_assets=0)
          (histories): HistoryManager()
          (random_seed): 9984043075503325450
          (max_epochs): 1
          (epoch): -1
          (iteration): -1
        )
    """

    def __init__(
        self,
        epoch: int = -1,
        iteration: int = -1,
        max_epochs: int = 1,
        random_seed: int = 9984043075503325450,
    ) -> None:
        self._epoch = epoch
        self._iteration = iteration
        self._max_epochs = max_epochs
        self._random_seed = random_seed

        self._histories = HistoryManager()
        self._modules = AssetManager()

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "modules": self._modules,
                    "histories": self._histories,
                    "random_seed": self._random_seed,
                    "max_epochs": self._max_epochs,
                    "epoch": self._epoch,
                    "iteration": self._iteration,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def random_seed(self) -> int:
        return self._random_seed

    def add_history(self, history: BaseHistory, key: str | None = None) -> None:
        self._histories.add_history(history=history, key=key)

    def add_module(self, name: str, module: Any) -> None:
        self._modules.add_asset(name=name, asset=module, replace_ok=True)

    def get_best_values(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        return self._histories.get_best_values(prefix, suffix)

    def get_history(self, key: str) -> BaseHistory:
        return self._histories.get_history(key)

    def get_histories(self) -> dict[str, BaseHistory]:
        return self._histories.get_histories()

    def get_module(self, name: str) -> Any:
        return self._modules.get_asset(name)

    def has_history(self, key: str) -> bool:
        return self._histories.has_history(key)

    def has_module(self, name: str) -> bool:
        return self._modules.has_asset(name)

    def increment_epoch(self, increment: int = 1) -> None:
        self._epoch += increment

    def increment_iteration(self, increment: int = 1) -> None:
        self._iteration += increment

    def load_state_dict(self, state_dict: dict) -> None:
        self._epoch = state_dict["epoch"]
        self._iteration = state_dict["iteration"]
        self._histories.load_state_dict(state_dict["histories"])
        self._modules.load_state_dict(state_dict["modules"])

    def remove_module(self, name: str) -> None:
        self._modules.remove_asset(name)

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "iteration": self._iteration,
            "histories": self._histories.state_dict(),
            "modules": self._modules.state_dict(),
        }
