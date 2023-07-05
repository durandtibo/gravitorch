from objectory import OBJECT_TARGET
from torch.utils.data.datapipes.map import SequenceWrapper

from gravitorch.datapipes.map import is_map_datapipe_config, setup_map_datapipe

############################################
#     Tests for is_map_datapipe_config     #
############################################


def test_is_map_datapipe_config_true() -> None:
    assert is_map_datapipe_config(
        {OBJECT_TARGET: "torch.utils.data.datapipes.map.SequenceWrapper", "sequence": [1, 2, 3, 4]}
    )


def test_is_map_datapipe_config_false() -> None:
    assert not is_map_datapipe_config({OBJECT_TARGET: "torch.nn.Identity"})


########################################
#     Tests for setup_map_datapipe     #
########################################


def test_setup_map_datapipe_object() -> None:
    datapipe = SequenceWrapper([1, 2, 3, 4])
    assert setup_map_datapipe(datapipe) is datapipe


def test_setup_map_datapipe_sequence() -> None:
    datapipe = setup_map_datapipe(
        {OBJECT_TARGET: "torch.utils.data.datapipes.map.SequenceWrapper", "sequence": [1, 2, 3, 4]}
    )
    assert isinstance(datapipe, SequenceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)
