__all__ = ["SourceIterDataPipe"]

import copy
import logging
from collections.abc import Iterable, Iterator

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_add_indent
from gravitorch.utils.summary import concise_summary

logger = logging.getLogger(__name__)


class SourceIterDataPipe(IterDataPipe):
    r"""Creates a simple source DataPipe from an iterable.

    Based on https://github.com/pytorch/pytorch/blob/3c2199b159b6ec57af3f7ea22d61ace9ce5cf5bc/torch/utils/data/datapipes/iter/utils.py#L8-L50  # noqa: B950

    Args:
        data (``iterable``): Specifies the input iterable.
        deepcopy (bool, optional): If ``True``, the input iterable
            object is deep-copied before to iterate over the data.
            It allows a deterministic behavior when in-place
            operations are performed on the data. Default: ``False``
    """

    def __init__(self, data: Iterable, deepcopy: bool = False):
        self._data = data
        self._deepcopy = bool(deepcopy)

    def __iter__(self) -> Iterator:
        source = self._data
        if self._deepcopy:
            try:
                source = copy.deepcopy(source)
            except TypeError:
                logger.warning(
                    "The input iterable can not be deepcopied, please be aware of in-place "
                    "modification would affect source data."
                )
        yield from source

    def __len__(self) -> int:
        try:
            return len(self._data)
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__qualname__} instance doesn't have valid length"
            ) from exc

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  deepcopy: {self._deepcopy},\n"
            f"  data:\n    {str_add_indent(concise_summary(self._data), num_spaces=4)}\n)"
        )
