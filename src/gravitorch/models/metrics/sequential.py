__all__ = ["SequentialMetric"]

from collections.abc import Sequence
from typing import Optional, Union

from torch import nn

from gravitorch.engines.base import BaseEngine
from gravitorch.models.metrics.base import BaseMetric, setup_metric


class SequentialMetric(BaseMetric):
    r"""Implements a "metric" to evaluate a sequence of metrics.

    This metric assumes that all the metrics are consistent. All the
    metrics should be compatible with the same inputs. The outputs
    are combined so the keys returned by each metric should be
    different.

    Args:
        metrics (sequence): Specifies the list of metrics to use or
            their configurations.
    """

    def __init__(self, metrics: Sequence[Union[BaseMetric, dict]]):
        super().__init__()
        self.metrics = nn.ModuleList([setup_metric(metric) for metric in metrics])

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the metrics to the provided engine.

        Note that each metric is attached individually so the metric
        handlers can be triggered at different events. This method
        calls the ``attach`` method of each metric. This method can
        be used to:

            - add event handler to the engine
            - set up history trackers

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        for metric in self.metrics:
            metric.attach(engine)

    def forward(self, *args, **kwargs) -> dict:
        r"""Updates the metrics given a mini-batch of examples.

        This method calls the ``forward`` method of each metric and
        combines the outputs in a single dictionary. All the metrics
        should be compatible with the inputs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: The outputs of the metrics.
        """
        outputs = {}
        for metric in self.metrics:
            out = metric(*args, **kwargs)
            if out:
                outputs.update(out)
        return outputs

    def reset(self) -> None:
        r"""Resets all the metrics."""
        for metric in self.metrics:
            metric.reset()

    def value(self, engine: Optional[BaseEngine] = None) -> dict:
        r"""Evaluates all the metrics and log the results given all the examples
        previously seen.

        This method calls the ``value`` method of each metric and
        combines the outputs in a single dictionary.

        Args:
            engine (``BaseEngine``, optional): Specifies the engine.
                This argument is required to log the results in the
                engine. Default: ``None``.

        Returns:
             dict: The results of the metric
        """
        outputs = {}
        for metric in self.metrics:
            outputs.update(metric.value(engine))
        return outputs
