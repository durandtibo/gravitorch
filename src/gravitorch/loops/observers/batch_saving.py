__all__ = ["PyTorchBatchSaver"]

import logging
from pathlib import Path
from typing import Any, Union

from gravitorch.engines.base import BaseEngine
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.utils.io import save_pytorch
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)


class PyTorchBatchSaver(BaseLoopObserver):
    r"""Implements a model batch saver to store batches in a PyTorch file
    (``torch.save``).

    Args:
        path (``pathlib.Path`` or str): Specifies the path where to
            store the examples.
        max_num_batches (int, optional): Specifies the maximum number
            of batches to save. Default: ``1000``
    """

    def __init__(self, path: Union[Path, str], max_num_batches: int = 1000):
        self._path = sanitize_path(path)
        self._max_num_batches = max_num_batches
        self._batches = []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(path={self._path}, "
            f"max_num_batches={self._max_num_batches:,})"
        )

    def update(self, engine: BaseEngine, model_input: Any, model_output: Any) -> None:
        r"""Update the observer.

        Args:
            engine (``BaseEngine``): Specifies the engine.
            model_input: Specifies a batch of model input.
            model_output: Specifies a batch of model output.
        """
        if len(self._batches) < self._max_num_batches:
            self._batches.append({"model_input": model_input, "model_output": model_output})

    def start(self, engine: BaseEngine) -> None:
        r"""Resets the model batch saver.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        self._batches.clear()

    def end(self, engine: BaseEngine) -> None:
        r"""Saves the batches in a PyTorch file.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        logger.info(f"Saving batches in {self._path}")
        save_pytorch(self._batches, self._path)
