from __future__ import annotations

__all__ = ["HypercubeVertexDataCreator", "create_hypercube_vertex"]

import logging

import torch
from arctix import summary
from torch import Tensor

from gravitorch import constants as ct
from gravitorch.data.datacreators.base import BaseDataCreator
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.seed import get_torch_generator

logger = logging.getLogger(__name__)


class HypercubeVertexDataCreator(BaseDataCreator[dict[str, Tensor]]):
    r"""Implements a data creator to create a toy classification dataset.

    The data are generated by using a hypercube. The targets are some
    vertices of the hypercube. Each input feature is a 1-hot
    representation of the target plus a Gaussian noise. These data can
    be used for a multi-class classification task.

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``1000``
        num_classes (int, optional): Specifies the number of classes.
            Default: 50
        feature_size (int, optional): Specifies the feature size.
            The feature size has to be greater than the number of
            classes. Default: ``64``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.2``
        random_seed (int, optional): Specifies the random seed used to
            initialize a ``torch.Generator`` object.
            Default: ``15782179921860610490``
        log_info (bool, optional): If ``True``, log information when
            the data are generated. Default: ``True``
    """

    def __init__(
        self,
        num_examples: int = 1000,
        num_classes: int = 50,
        feature_size: int = 64,
        noise_std: float = 0.2,
        random_seed: int = 15782179921860610490,
        log_info: bool = True,
    ) -> None:
        if num_examples < 1:
            raise ValueError(f"The number of examples ({num_examples}) has to be greater than 0")
        self._num_examples = int(num_examples)

        if num_classes < 1:
            raise ValueError(f"The number of classes ({num_classes}) has to be greater than 0")
        self._num_classes = int(num_classes)

        if feature_size < num_classes:
            raise ValueError(
                f"The feature dimension ({feature_size:,}) has to be greater or equal to the "
                f"number of classes ({num_classes:,})"
            )
        self._feature_size = int(feature_size)

        if noise_std < 0:
            raise ValueError(
                f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
                "greater or equal than 0"
            )
        self._noise_std = float(noise_std)
        self._log_info = bool(log_info)

        self._generator = get_torch_generator(random_seed)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"num_examples={self._num_examples:,}, "
            f"num_classes={self._num_classes:,}, "
            f"feature_size={self._feature_size:,}, "
            f"noise_std={self._noise_std:,}, "
            f"random_seed={self.random_seed})"
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # torch.Generator cannot be serialized but its state can.
        state["_generator"] = state["_generator"].get_state()
        return state

    def __setstate__(self, state: dict) -> None:
        # Recreate the torch.Generator because only its state was serialized
        generator = torch.Generator()
        state["_generator"] = generator.set_state(state["_generator"])
        self.__dict__.update(state)

    @property
    def num_examples(self) -> int:
        r"""``int``: The number of examples when the data are
        created."""
        return self._num_examples

    @property
    def num_classes(self) -> int:
        r"""``int``: The number of classes when the data are created."""
        return self._num_classes

    @property
    def feature_size(self) -> int:
        r"""``int``: The feature size when the data are created."""
        return self._feature_size

    @property
    def noise_std(self) -> float:
        r"""``float``: The standard deviation of the Gaussian noise."""
        return self._noise_std

    @property
    def random_seed(self) -> int:
        r"""int: The random seed used to initialize a ``torch.Generator`` object."""
        return self._generator.initial_seed()

    def create(self, engine: BaseEngine | None = None) -> dict[str, Tensor]:
        r"""Creates data.

        Args:
        ----
            engine (``BaseEngine`` or ``None``): Specifies an engine.
                This input is not used in this data creator.
                Default: ``None``

        Returns:
        -------
            dict: A dictionary with two keys:
                - ``'input'``: a ``torch.Tensor`` of type float and
                    shape ``(num_examples, feature_size)``. This
                    tensor represents the input features.
                - ``'target'``: a ``torch.Tensor`` of type long and
                    shape ``(num_examples,)``. This tensor represents
                    the targets.
        """
        if self._log_info:
            logger.info(f"Creating {self.num_examples:,} examples (seed={self.random_seed})...")
        data = create_hypercube_vertex(
            num_examples=self._num_examples,
            num_classes=self._num_classes,
            feature_size=self._feature_size,
            noise_std=self._noise_std,
            generator=self._generator,
        )
        if self._log_info:
            logger.info(f"Created data\n{summary(data)}")
        return data


def create_hypercube_vertex(
    num_examples: int = 1000,
    num_classes: int = 50,
    feature_size: int = 64,
    noise_std: float = 0.2,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    r"""Create a toy classification dataset based on hypercube vertex
    structure.

    The data are generated by using a hypercube. The targets are some
    vertices of the hypercube. Each input feature is a 1-hot
    representation of the target plus a Gaussian noise. These data can
    be used for a multi-class classification task.

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``1000``
        num_classes (int, optional): Specifies the number of classes.
            Default: 50
        feature_size (int, optional): Specifies the feature size.
            The feature size has to be greater than the number of
            classes. Default: ``64``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.2``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator. Default: ``None``

    Returns:
    -------
        dict: A dictionary with two keys:
            - ``'input'``: a ``torch.Tensor`` of type float and
                shape ``(num_examples, feature_size)``. This
                tensor represents the input features.
            - ``'target'``: a ``torch.Tensor`` of type long and
                shape ``(num_examples,)``. This tensor represents
                the targets.
    """

    if num_examples < 1:
        raise RuntimeError(f"The number of examples ({num_examples}) has to be greater than 0")
    if num_classes < 1:
        raise RuntimeError(f"The number of classes ({num_classes}) has to be greater than 0")
    if feature_size < num_classes:
        raise RuntimeError(
            f"The feature dimension ({feature_size:,}) has to be greater or equal to the "
            f"number of classes ({num_classes:,})"
        )
    if noise_std < 0:
        raise RuntimeError(
            f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
            "greater or equal than 0"
        )

    # Generate the target of each example.
    targets = torch.randint(0, num_classes, (num_examples,), generator=generator)
    # Generate the features. Each class should be a vertex of the hyper-cube
    # plus Gaussian noise.
    features = torch.randn(num_examples, feature_size, generator=generator).mul(noise_std)
    features.scatter_add_(1, targets.view(num_examples, 1), torch.ones(num_examples, 1))

    return {ct.TARGET: targets, ct.INPUT: features}
