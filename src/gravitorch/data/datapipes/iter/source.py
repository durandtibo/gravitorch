__all__ = ["SourceWrapperIterDataPipe"]

import copy
import logging
from collections.abc import Iterable, Iterator

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_indent
from gravitorch.utils.summary import concise_summary

logger = logging.getLogger(__name__)


class SourceWrapperIterDataPipe(IterDataPipe):
    r"""Creates a simple source DataPipe from an iterable.

    Based on https://github.com/pytorch/pytorch/blob/3c2199b159b6ec57af3f7ea22d61ace9ce5cf5bc/torch/utils/data/datapipes/iter/utils.py#L8-L50  # noqa: B950

    Args:
        source (``iterable``): Specifies the input iterable.
        deepcopy (bool, optional): If ``True``, the input iterable
            object is deep-copied before to iterate over the data.
            It allows a deterministic behavior when in-place
            operations are performed on the data. Default: ``False``
    """

    def __init__(self, source: Iterable, deepcopy: bool = False):
        self._source = source
        self._deepcopy = bool(deepcopy)

    def __iter__(self) -> Iterator:
        source = self._source
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
        return len(self._source)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  deepcopy: {self._deepcopy},\n"
            f"  source:\n    {str_indent(concise_summary(self._source), num_spaces=4)}\n)"
        )
