"""This module defines a simple model which is composed of 3 modules:

network, criterion and metrics.
"""

__all__ = ["VanillaModel"]

import logging
from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.nn.utils.rnn import PackedSequence

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.models.base import BaseModel
from gravitorch.models.metrics import BaseMetric
from gravitorch.models.utils import attach_module_to_engine
from gravitorch.nn.utils.factory import setup_module
from gravitorch.nn.utils.state_dict import load_checkpoint_to_module
from gravitorch.utils.seed import manual_seed

logger = logging.getLogger(__name__)


class VanillaModel(BaseModel):
    r"""Implements a simple model which is composed of 3 modules: network,
    criterion and metrics.

    The assumptions of the ``VanillaModel`` are:

        - the input of the model is a dict containing the data of the
            mini-batch.
        - the network follow the ``BaseNetwork`` API.
        - the input of the criterion is the dicts containing the
            network output and the mini-batch data.
        - the criterion should return a dictionary containing the
            loss.
        - the metrics are optional, or it is possible to define a
            metric only for train or eval.
        - the input of the metric is the dicts containing the
            criterion output, network output and the mini-batch data.
        - the keys returned by the network, criterion and metrics must
            be different.

    Note that the metric names are ``'train_metric'`` and
    ``'eval_metric'`` because it is not possible to use
    ``'train'`` and ``'eval'`` because they are already used by
    PyTorch.

    Args:
        network (``torch.nn.Module`` or dict): Specifies the network
            module or its configuration.
        criterion (``torch.nn.Module`` or dict): Specifies the
            criterion module or its configuration.
        metrics (``torch.nn.ModuleDict`` or dict): Specifies the
            metrics or their configuration.
        checkpoint_path (``pathlib.Path`` or str or ``None``):
            Specifies a path to a model checkpoint. This weights in
            the checkpoint are used to initialize the model. ``None``
             means there is no checkpoint to load. Default: ``None``.
        random_seed (int, optional): Specifies the random seed.
            Default: ``6671429959452193306``
    """

    def __init__(
        self,
        network: Union[Module, dict],
        criterion: Union[Module, dict],
        metrics: Union[ModuleDict, dict, None] = None,
        checkpoint_path: Union[Path, str, None] = None,
        random_seed: int = 6671429959452193306,
    ):
        super().__init__()
        manual_seed(random_seed)  # Fix the random seed for reproducibility purpose
        self.network = setup_module(network)
        logger.info(f"network:\n{self.network}")
        self.criterion = setup_module(criterion)
        logger.info(f"criterion:\n{self.criterion}")
        self.metrics = self._setup_metrics(metrics)
        logger.info(f"metrics:\n{self.metrics}")

        if checkpoint_path:
            load_checkpoint_to_module(checkpoint_path, self)

    def forward(self, batch: dict) -> dict:
        """Prepares the batch and feed it to the network, criterion and metric.

        Args:
            batch (dict): Specifies the dict containing data: inputs,
                targets, etc.

        Returns:
            out (dict): a dictionary of outputs including the loss value
        """
        net_out = self._parse_net_out(
            self.network(*tuple(batch[key] for key in self.network.get_input_names()))
        )
        cri_out = self.criterion(net_out, batch)
        met_out = self._get_metric_out(cri_out, net_out, batch)
        return self._get_model_out(net_out, cri_out, met_out)

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches current model to the provided engine.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        attach_module_to_engine(self.network, engine)
        attach_module_to_engine(self.criterion, engine)
        for metric in self.metrics.values():
            attach_module_to_engine(metric, engine)

    def _setup_metrics(self, metrics: Union[ModuleDict, dict, None]) -> ModuleDict:
        r"""Sets up the metrics.

        Args:
            metrics (``torch.nn.ModuleDict`` or dict): Specifies the
                metrics or their configuration.

        Returns:
            ``torch.nn.ModuleDict``: The metrics
        """
        if isinstance(metrics, ModuleDict):
            return metrics

        metrics = metrics or {}
        for key, metric in metrics.items():
            if isinstance(metric, dict):
                metric = BaseMetric.factory(**metric)
            metrics[key] = metric
        return ModuleDict(metrics)

    def _parse_net_out(self, net_out: Union[Tensor, PackedSequence, list, tuple]) -> dict:
        r"""Parses the network outputs to a dict.

        Args:
            net_out: Specifies the network output.

        Returns:
            dict: The dictionary containing the network output.
        """
        if torch.is_tensor(net_out) or isinstance(net_out, PackedSequence):
            net_out = (net_out,)
        return {
            output_key: net_out[i] for i, output_key in enumerate(self.network.get_output_names())
        }

    def _get_metric_out(self, cri_out: dict, net_out: dict, batch: dict) -> dict:
        r"""Computes the metric output.

        Args:
            cri_out (dict): Specifies the dict with the criterion
                output.
            net_out (dict): Specifies the dict with the network
                output.
            batch (dict): Specifies the dict containing data: inputs,
                targets, etc.

        Returns:
            dict: The dictionary containing the metric values. The
                dictionary can be empty.
        """
        # It is not possible to use 'train' or 'eval' because these
        # names are already used in ``torch.nn.Module``
        mode = f"{ct.TRAIN if self.training else ct.EVAL}_metric"
        met_out = {}
        if mode in self.metrics:
            met_out = self.metrics[mode](cri_out, net_out, batch)
        return met_out or {}

    def _get_model_out(self, net_out: dict, cri_out: dict, met_out: dict) -> dict:
        r"""Computes the model output by combining the outputs of the network,
        criterion and metric.

        Note that the keys have to be unique otherwise only one will
        be returned by the model.

        Args:
            net_out (dict): Specifies the dict with the network
                output.
            cri_out (dict): Specifies the dict with the criterion
                output.
            met_out (dict): Specifies the dict containing data:
                inputs, targets, etc.

        Returns:
            dict: The dictionary containing the model outputs.
        """
        out = {}
        out.update(net_out)
        out.update(cri_out)
        out.update(met_out)
        return out
