__all__ = ["WeightedSumLoss"]

from typing import Optional, Union

from torch.nn import Module, ModuleDict

from gravitorch import constants as ct
from gravitorch.nn import setup_module


class WeightedSumLoss(Module):
    r"""Implements a loss function (a.k.a. criterion) that computes the weighted
    sum of multiple loss functions.

    Args:
        criteria (``torch.nn.ModuleDict`` or dict): Specifies the
            loss functions (a.k.a. criteria)
            or their configuration.
        weights (dict or ``None``, optional): Specifies the weight
            associated to each loss function. The keys should match
            with the criteria keys. By default, the weight is ``1.0``
            if it is not specified. Default: ``None``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.models.criteria import WeightedSumLoss
        # Example without specified weight. Weights are initialized to 1.0
        >>> criterion = WeightedSumLoss(
        ...     criteria={
        ...         "value": {'_target_': "torch.nn.MSELoss"},
        ...         "time": {'_target_': "torch.nn.L1Loss"},
        ...     },
        ... )
        >>> criterion
        WeightedSumLoss(
          weights: {'value': 1.0, 'time': 1.0}
          (criteria): ModuleDict(
            (value): MSELoss()
            (time): L1Loss()
          )
        )
        # Example with specified weight.
        >>> criterion = WeightedSumLoss(
        ...     criteria={
        ...         "value": {'_target_': "torch.nn.MSELoss"},
        ...         "time": {'_target_': "torch.nn.L1Loss"},
        ...     },
        ...     weights={"value": 2.0, "time": 0.5},
        ... )
        >>> criterion
        WeightedSumLoss(
          weights: {'value': 2.0, 'time': 0.5}
          (criteria): ModuleDict(
            (value): MSELoss()
            (time): L1Loss()
          )
        )
        # Example with partially specified weight.
        >>> criterion = WeightedSumLoss(
        ...     criteria={
        ...         "value": {'_target_': "torch.nn.MSELoss"},
        ...         "time": {'_target_': "torch.nn.L1Loss"},
        ...     },
        ...     weights={"time": 0.5},
        ... )
        >>> criterion
        WeightedSumLoss(
          weights: {'value': 1.0, 'time': 0.5}
          (criteria): ModuleDict(
            (value): MSELoss()
            (time): L1Loss()
          )
        )
    """

    def __init__(
        self,
        criteria: Union[ModuleDict, dict[str, Union[Module, dict]]],
        weights: Optional[dict[str, Union[int, float]]] = None,
    ):
        super().__init__()
        if not isinstance(criteria, ModuleDict):
            criteria = ModuleDict({key: setup_module(value) for key, value in criteria.items()})
        self.criteria = criteria
        self._weights = weights or {}
        for key in self.criteria.keys():
            self._weights[key] = self._weights.get(key, 1.0)

    def extra_repr(self) -> str:
        return f"weights: {self._weights}"

    def forward(self, *args, **kwargs) -> dict:
        out = {ct.LOSS: 0.0}
        for key, criterion in self.criteria.items():
            loss = criterion(*args, **kwargs)
            if isinstance(loss, dict):
                loss = loss[ct.LOSS]
            out[f"{ct.LOSS}_{key}"] = loss * self._weights[key]
            out[ct.LOSS] += out[f"{ct.LOSS}_{key}"]
        return out
