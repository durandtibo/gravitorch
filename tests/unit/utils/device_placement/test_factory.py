from objectory import OBJECT_TARGET

from gravitorch.utils.device_placement import (
    CpuDevicePlacement,
    NoOpDevicePlacement,
    setup_device_placement,
)

############################################
#     Tests for setup_device_placement     #
############################################


def test_setup_device_placement_none() -> None:
    assert isinstance(setup_device_placement(None), NoOpDevicePlacement)


def test_setup_device_placement_object() -> None:
    device_placement = CpuDevicePlacement()
    assert setup_device_placement(device_placement) is device_placement


def test_setup_device_placement_dict() -> None:
    assert isinstance(
        setup_device_placement(
            {OBJECT_TARGET: "gravitorch.utils.device_placement.CpuDevicePlacement"}
        ),
        CpuDevicePlacement,
    )
