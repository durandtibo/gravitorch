import logging

import torch
from coola import objects_are_equal
from pytest import LogCaptureFixture, raises
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.map import SequenceWrapper

from gravitorch.datapipes import clone_datapipe
from gravitorch.testing import torchdata_available

####################################
#     Tests for clone_datapipe     #
####################################


def test_clone_datapipe() -> None:
    datapipe = clone_datapipe(IterableWrapper([torch.full((3,), 1.0) for _ in range(5)]))
    assert objects_are_equal(
        [x.add_(1) for x in datapipe], [torch.full((3,), 2.0) for _ in range(5)]
    )
    assert objects_are_equal([torch.full((3,), 1.0) for _ in range(5)], list(datapipe))


def test_clone_datapipe_iter() -> None:
    datapipe = IterableWrapper([torch.full((3,), 1.0) for _ in range(5)])
    dp = clone_datapipe(datapipe)
    assert isinstance(dp, IterableWrapper)
    assert dp is not datapipe
    assert objects_are_equal(tuple(dp), tuple(datapipe))


@torchdata_available
def test_clone_datapipe_map() -> None:
    datapipe = SequenceWrapper([torch.full((3,), 1.0) for _ in range(5)])
    dp = clone_datapipe(datapipe)
    assert isinstance(dp, SequenceWrapper)
    assert dp is not datapipe
    assert objects_are_equal(tuple(dp.to_iterdatapipe()), tuple(datapipe.to_iterdatapipe()))


def test_clone_datapipe_cannot_be_copied() -> None:
    datapipe = IterableWrapper(torch.full((3,), 1.0) for _ in range(5))
    with raises(TypeError, match="The DataPipe can not be cloned"):
        clone_datapipe(datapipe)


def test_clone_datapipe_cannot_be_copied_raise_error_false(caplog: LogCaptureFixture) -> None:
    datapipe = IterableWrapper(torch.full((3,), 1.0) for _ in range(5))
    with caplog.at_level(level=logging.WARNING):
        dp = clone_datapipe(datapipe, raise_error=False)
        assert dp is datapipe
        assert caplog.messages[0].startswith("The DataPipe can not be cloned")
