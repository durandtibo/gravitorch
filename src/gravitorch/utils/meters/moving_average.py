__all__ = ["ExponentialMovingAverage", "MovingAverage"]

from collections import deque
from collections.abc import Iterable
from typing import Any, Union

import torch

from gravitorch.utils.meters.exceptions import EmptyMeterError


class MovingAverage:
    r"""Defines a class to compute and store the exponential moving average
    value of float number.

    Args:
        values (iterable, optional): Specifies the initial values.
            Default: ``tuple()``
        window_size (int, optional): Specifies the maximum window
            size. Default: ``20``
    """

    def __init__(self, values: Iterable[Union[int, float]] = (), window_size: int = 20):
        self._deque = deque(values, maxlen=window_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(window_size={self.window_size:,})"

    @property
    def values(self) -> tuple[Union[int, float]]:
        return tuple(self._deque)

    @property
    def window_size(self) -> int:
        return self._deque.maxlen

    def clone(self) -> "MovingAverage":
        r"""Creates a copy of the current meter.

        Returns:
            ``MovingAverage``: A copy of the current meter.
        """
        return MovingAverage(values=tuple(self._deque), window_size=self.window_size)

    def equal(self, other) -> bool:
        r"""Indicates if two meters are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the meters are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, MovingAverage):
            return False
        return self.state_dict() == other.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a state to the history tracker.

        Args:
            state_dict (dict): Specifies a dictionary containing state
                keys with values.
        """
        self._deque = deque(state_dict["values"], maxlen=state_dict["window_size"])

    def reset(self) -> None:
        r"""Reset the meter."""
        self._deque.clear()

    def smoothed_average(self) -> float:
        r"""Computes the smoothed average value.

        Returns:
            float: The smoothed average value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._deque:
            raise EmptyMeterError("The moving average meter is empty")
        return torch.tensor(self.values).float().mean().item()

    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: The state values in a dict.
        """
        return {"values": self.values, "window_size": self.window_size}

    def update(self, value: Union[int, float]) -> None:
        r"""Updates the meter given a new value.

        Args:
            value (int or float): Specifies the value to add to the
                meter.
        """
        self._deque.append(value)


class ExponentialMovingAverage:
    r"""Defines a class to compute and store the exponential moving average
    value of float number.

    Args:
        alpha (float, optional): Specifies the smoothing factor such
            as ``0 < alpha < 1``.
        count (int, optional): Specifies the initial count value.
            Default: ``0``
        smoothed_average (float, optional): Specifies the initial
            value of the smoothed average. Default: ``0.0``
    """

    def __init__(
        self,
        alpha: float = 0.98,
        count: int = 0,
        smoothed_average: float = 0.0,
    ):
        self._alpha = float(alpha)
        self._count = int(count)
        self._smoothed_average = float(smoothed_average)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(alpha={self._alpha}, count={self._count:,}, "
            f"smoothed_average={self._smoothed_average}, )"
        )

    @property
    def count(self) -> int:
        r"""int: The number of examples in the meter since the last
        reset."""
        return self._count

    def clone(self) -> "ExponentialMovingAverage":
        r"""Creates a copy of the current meter.

        Returns:
            ``ExponentialMovingAverage``: A copy of the current meter.
        """
        return ExponentialMovingAverage(
            alpha=self._alpha,
            count=self._count,
            smoothed_average=self._smoothed_average,
        )

    def equal(self, other) -> bool:
        r"""Indicates if two meters are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the meters are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, ExponentialMovingAverage):
            return False
        return self.state_dict() == other.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a state to the history tracker.

        Args:
            state_dict (dict): Specifies a dictionary containing state
                keys with values.
        """
        self._alpha = state_dict["alpha"]
        self._count = state_dict["count"]
        self._smoothed_average = state_dict["smoothed_average"]

    def reset(self) -> None:
        r"""Reset the meter."""
        self._count = 0
        self._smoothed_average = 0.0

    def smoothed_average(self) -> float:
        r"""Computes the smoothed average value.

        Returns:
            float: The smoothed average value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The exponential moving average meter is empty")
        return self._smoothed_average

    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: The state values in a dict.
        """
        return {
            "alpha": self._alpha,
            "count": self._count,
            "smoothed_average": self._smoothed_average,
        }

    def update(self, value: float) -> None:
        r"""Updates the meter given a new value.

        Args:
            value (float): Specifies the value to add to the meter.
        """
        self._smoothed_average = self._alpha * self._smoothed_average + (1.0 - self._alpha) * value
        self._count += 1
