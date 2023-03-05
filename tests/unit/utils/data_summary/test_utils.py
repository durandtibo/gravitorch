from objectory import OBJECT_TARGET

from gravitorch.utils.data_summary import (
    FloatDataSummary,
    NoOpDataSummary,
    setup_data_summary,
)

########################################
#     Tests for setup_data_summary     #
########################################


def test_setup_data_summary_object() -> None:
    data_summary = FloatDataSummary()
    assert setup_data_summary(data_summary) is data_summary


def test_setup_data_summary_dict() -> None:
    assert isinstance(
        setup_data_summary({OBJECT_TARGET: "gravitorch.utils.data_summary.FloatDataSummary"}),
        FloatDataSummary,
    )


def test_setup_data_summary_none() -> None:
    assert isinstance(setup_data_summary(None), NoOpDataSummary)
