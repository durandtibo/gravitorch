from objectory import OBJECT_TARGET
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.map import SequenceWrapper

from gravitorch.datapipes import setup_datapipe

####################################
#     Tests for setup_datapipe     #
####################################


def test_setup_datapipe_object_iter() -> None:
    datapipe = IterableWrapper((1, 2, 3, 4, 5))
    assert setup_datapipe(datapipe) is datapipe


def test_setup_datapipe_object_map() -> None:
    datapipe = SequenceWrapper((1, 2, 3, 4, 5))
    assert setup_datapipe(datapipe) is datapipe


def test_setup_datapipe_config_iter() -> None:
    assert isinstance(
        setup_datapipe(
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": (1, 2, 3, 4, 5),
            }
        ),
        IterableWrapper,
    )


def test_setup_datapipe_config_map() -> None:
    assert isinstance(
        setup_datapipe(
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.map.SequenceWrapper",
                "sequence": (1, 2, 3, 4, 5),
            }
        ),
        SequenceWrapper,
    )
