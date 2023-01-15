from typing import Union

import torch
from coola import objects_are_allclose
from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch import Tensor

from gravitorch.models.metrics import EmptyMetricError
from gravitorch.models.metrics.state import (
    AccuracyState,
    ErrorState,
    ExtendedAccuracyState,
    ExtendedErrorState,
    MeanErrorState,
    RootMeanErrorState,
    setup_state,
)
from gravitorch.utils.history import MaxScalarHistory, MinScalarHistory

####################################
#     Tests for MeanErrorState     #
####################################


def test_mean_error_state_str():
    assert str(MeanErrorState()).startswith("MeanErrorState(")


def test_mean_error_state_get_histories():
    histories = MeanErrorState().get_histories()
    assert len(histories) == 1
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == "mean"


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_mean_error_state_get_histories_prefix_suffix(prefix: str, suffix: str):
    histories = MeanErrorState().get_histories(prefix, suffix)
    assert len(histories) == 1
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == f"{prefix}mean{suffix}"


def test_mean_error_state_reset():
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6
    state.reset()
    assert state.num_predictions == 0


def test_mean_error_state_update_1d():
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0


def test_mean_error_state_update_2d():
    state = MeanErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0


def test_mean_error_state_value():
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state.value() == {"mean": 2.5, "num_predictions": 6}


def test_mean_error_state_value_correct():
    state = MeanErrorState()
    state.update(torch.zeros(4))
    assert state.value() == {"mean": 0.0, "num_predictions": 4}


def test_mean_error_state_value_track_num_predictions_false():
    state = MeanErrorState(track_num_predictions=False)
    state.update(torch.arange(6))
    assert state.value() == {"mean": 2.5}


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_mean_error_state_value_prefix_suffix(prefix: str, suffix: str):
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state.value(prefix, suffix) == {
        f"{prefix}mean{suffix}": 2.5,
        f"{prefix}num_predictions{suffix}": 6,
    }


def test_mean_error_state_value_empty():
    state = MeanErrorState()
    with raises(EmptyMetricError):
        state.value()


###############################################
#     Tests for RootMeanErrorState     #
###############################################


def test_root_mean_error_state_str():
    assert str(RootMeanErrorState()).startswith("RootMeanErrorState(")


def test_root_mean_error_state_get_histories():
    histories = RootMeanErrorState().get_histories()
    assert len(histories) == 1
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == "root_mean"


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_root_mean_error_state_get_histories_prefix_suffix(prefix: str, suffix: str):
    histories = RootMeanErrorState().get_histories(prefix, suffix)
    assert len(histories) == 1
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == f"{prefix}root_mean{suffix}"


def test_root_mean_error_state_reset():
    state = RootMeanErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6
    state.reset()
    assert state.num_predictions == 0


def test_root_mean_error_state_update_1d():
    state = RootMeanErrorState()
    state.update(torch.arange(6))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0


def test_root_mean_error_state_update_2d():
    state = RootMeanErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0


def test_root_mean_error_state_value():
    state = RootMeanErrorState()
    state.update(torch.tensor([1, 9, 2, 7, 3, 2]))
    assert state.value() == {"root_mean": 2.0, "num_predictions": 6}


def test_root_mean_error_state_value_correct():
    state = RootMeanErrorState()
    state.update(torch.zeros(4))
    assert state.value() == {"root_mean": 0.0, "num_predictions": 4}


def test_root_mean_error_state_value_track_num_predictions_false():
    state = RootMeanErrorState(track_num_predictions=False)
    state.update(torch.tensor([1, 9, 2, 7, 3, 2]))
    assert state.value() == {"root_mean": 2.0}


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_root_mean_error_state_value_prefix_suffix(prefix: str, suffix: str):
    state = RootMeanErrorState()
    state.update(torch.tensor([1, 9, 2, 7, 3, 2]))
    assert state.value(prefix, suffix) == {
        f"{prefix}root_mean{suffix}": 2.0,
        f"{prefix}num_predictions{suffix}": 6,
    }


def test_root_mean_error_state_value_empty():
    state = RootMeanErrorState()
    with raises(EmptyMetricError):
        state.value()


################################
#     Tests for ErrorState     #
################################


def test_error_state_str():
    assert str(ErrorState()).startswith("ErrorState(")


def test_error_state_get_histories():
    histories = ErrorState().get_histories()
    assert len(histories) == 4
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == "mean"
    assert isinstance(histories[1], MinScalarHistory)
    assert histories[1].name == "min"
    assert isinstance(histories[2], MinScalarHistory)
    assert histories[2].name == "max"
    assert isinstance(histories[3], MinScalarHistory)
    assert histories[3].name == "sum"


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_error_state_get_histories_prefix_suffix(prefix: str, suffix: str):
    histories = ErrorState().get_histories(prefix, suffix)
    assert len(histories) == 4
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == f"{prefix}mean{suffix}"
    assert isinstance(histories[1], MinScalarHistory)
    assert histories[1].name == f"{prefix}min{suffix}"
    assert isinstance(histories[2], MinScalarHistory)
    assert histories[2].name == f"{prefix}max{suffix}"
    assert isinstance(histories[3], MinScalarHistory)
    assert histories[3].name == f"{prefix}sum{suffix}"


def test_error_state_reset():
    state = ErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6
    state.reset()
    assert state.num_predictions == 0


def test_error_state_update_1d():
    state = ErrorState()
    state.update(torch.arange(6))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0
    assert state._meter.max() == 5.0
    assert state._meter.min() == 0.0


def test_error_state_update_2d():
    state = ErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0
    assert state._meter.max() == 5.0
    assert state._meter.min() == 0.0


def test_error_state_value():
    state = ErrorState()
    state.update(torch.arange(6))
    assert state.value() == {
        "mean": 2.5,
        "min": 0.0,
        "max": 5.0,
        "sum": 15.0,
        "num_predictions": 6,
    }


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_error_state_value_prefix_suffix(prefix: str, suffix: str):
    state = ErrorState()
    state.update(torch.arange(6))
    assert state.value(prefix, suffix) == {
        f"{prefix}mean{suffix}": 2.5,
        f"{prefix}min{suffix}": 0.0,
        f"{prefix}max{suffix}": 5.0,
        f"{prefix}sum{suffix}": 15.0,
        f"{prefix}num_predictions{suffix}": 6,
    }


def test_error_state_value_empty():
    state = ErrorState()
    with raises(EmptyMetricError):
        state.value()


########################################
#     Tests for ExtendedErrorState     #
########################################


def test_extended_error_state_str():
    assert str(ExtendedErrorState()).startswith("ExtendedErrorState(")


@mark.parametrize("quantiles", (torch.tensor([0.5, 0.9]), [0.5, 0.9], (0.5, 0.9)))
def test_extended_error_state_init_quantiles(quantiles: Union[Tensor, list, tuple]):
    assert ExtendedErrorState(quantiles)._quantiles.equal(
        torch.tensor([0.5, 0.9], dtype=torch.float)
    )


def test_extended_error_state_init_quantiles_empty():
    assert ExtendedErrorState()._quantiles.equal(torch.tensor([]))


def test_extended_error_state_get_histories_no_quantile():
    histories = ExtendedErrorState().get_histories()
    assert len(histories) == 5
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == "mean"
    assert isinstance(histories[1], MinScalarHistory)
    assert histories[1].name == "median"
    assert isinstance(histories[2], MinScalarHistory)
    assert histories[2].name == "min"
    assert isinstance(histories[3], MinScalarHistory)
    assert histories[3].name == "max"
    assert isinstance(histories[4], MinScalarHistory)
    assert histories[4].name == "sum"


def test_extended_error_state_get_histories_quantiles():
    histories = ExtendedErrorState(quantiles=[0.5, 0.9]).get_histories()
    assert len(histories) == 7
    assert isinstance(histories[0], MinScalarHistory)
    assert histories[0].name == "mean"
    assert isinstance(histories[1], MinScalarHistory)
    assert histories[1].name == "median"
    assert isinstance(histories[2], MinScalarHistory)
    assert histories[2].name == "min"
    assert isinstance(histories[3], MinScalarHistory)
    assert histories[3].name == "max"
    assert isinstance(histories[4], MinScalarHistory)
    assert histories[4].name == "sum"
    assert isinstance(histories[5], MinScalarHistory)
    assert histories[5].name == "quantile_0.5"
    assert isinstance(histories[6], MinScalarHistory)
    assert histories[6].name == "quantile_0.9"


def test_extended_error_state_reset():
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6
    state.reset()
    assert state.num_predictions == 0


def test_extended_error_state_update_1d():
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6


def test_extended_error_state_update_2d():
    state = ExtendedErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state.num_predictions == 6


def test_extended_error_state_value_no_quantiles():
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert objects_are_allclose(
        state.value(),
        {
            "mean": 2.5,
            "median": 2,
            "min": 0,
            "max": 5,
            "sum": 15,
            "std": 1.8708287477493286,
            "num_predictions": 6,
        },
    )


def test_extended_error_state_value_with_quantiles():
    state = ExtendedErrorState(quantiles=[0.5, 0.9])
    state.update(torch.arange(11))
    assert objects_are_allclose(
        state.value(),
        {
            "mean": 5.0,
            "median": 5,
            "min": 0,
            "max": 10,
            "sum": 55,
            "std": 3.316624879837036,
            "quantile_0.5": 5.0,
            "quantile_0.9": 9.0,
            "num_predictions": 11,
        },
    )


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_extended_error_state_value_prefix_suffix(prefix: str, suffix: str):
    state = ExtendedErrorState(quantiles=[0.5, 0.9])
    state.update(torch.arange(11))
    assert objects_are_allclose(
        state.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 5.0,
            f"{prefix}median{suffix}": 5,
            f"{prefix}min{suffix}": 0,
            f"{prefix}max{suffix}": 10,
            f"{prefix}sum{suffix}": 55,
            f"{prefix}std{suffix}": 3.316624879837036,
            f"{prefix}quantile_0.5{suffix}": 5.0,
            f"{prefix}quantile_0.9{suffix}": 9.0,
            f"{prefix}num_predictions{suffix}": 11,
        },
        show_difference=True,
    )


def test_extended_error_state_value_empty():
    state = ExtendedErrorState()
    with raises(EmptyMetricError):
        state.value()


###################################
#     Tests for AccuracyState     #
###################################


def test_accuracy_state_str():
    assert str(AccuracyState()).startswith("AccuracyState(")


def test_accuracy_state_get_histories():
    histories = AccuracyState().get_histories()
    assert len(histories) == 1
    assert isinstance(histories[0], MaxScalarHistory)
    assert histories[0].name == "accuracy"


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_accuracy_state_get_histories_prefix_suffix(prefix: str, suffix: str):
    histories = AccuracyState().get_histories(prefix, suffix)
    assert len(histories) == 1
    assert isinstance(histories[0], MaxScalarHistory)
    assert histories[0].name == f"{prefix}accuracy{suffix}"


def test_accuracy_state_reset():
    state = AccuracyState()
    state.update(torch.eye(2))
    assert state.num_predictions == 4
    state.reset()
    assert state.num_predictions == 0


def test_accuracy_state_update_1d():
    state = AccuracyState()
    state.update(torch.ones(4))
    assert state._meter.count == 4
    assert state._meter.sum() == 4.0


def test_accuracy_state_update_2d():
    state = AccuracyState()
    state.update(torch.ones(2, 3))
    assert state._meter.count == 6
    assert state._meter.sum() == 6.0


def test_accuracy_state_value_correct():
    state = AccuracyState()
    state.update(torch.ones(4))
    assert state.value() == {"accuracy": 1.0, "num_predictions": 4}


def test_accuracy_state_value_partially_correct():
    state = AccuracyState()
    state.update(torch.eye(2))
    assert state.value() == {"accuracy": 0.5, "num_predictions": 4}


def test_accuracy_state_value_incorrect():
    state = AccuracyState()
    state.update(torch.zeros(4))
    assert state.value() == {"accuracy": 0.0, "num_predictions": 4}


def test_accuracy_state_value_track_num_predictions_false():
    state = AccuracyState(track_num_predictions=False)
    state.update(torch.eye(2))
    assert state.value() == {"accuracy": 0.5}


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_accuracy_state_value_prefix_suffix(prefix: str, suffix: str):
    state = AccuracyState()
    state.update(torch.eye(2))
    assert state.value(prefix, suffix) == {
        f"{prefix}accuracy{suffix}": 0.5,
        f"{prefix}num_predictions{suffix}": 4,
    }


def test_accuracy_state_value_empty():
    state = AccuracyState()
    with raises(EmptyMetricError):
        state.value()


###########################################
#     Tests for ExtendedAccuracyState     #
###########################################


def test_extended_accuracy_state_str():
    assert str(ExtendedAccuracyState()).startswith("ExtendedAccuracyState(")


def test_extended_accuracy_state_get_histories():
    histories = ExtendedAccuracyState().get_histories()
    assert len(histories) == 4
    assert isinstance(histories[0], MaxScalarHistory)
    assert histories[0].name == "accuracy"
    assert isinstance(histories[1], MinScalarHistory)
    assert histories[1].name == "error"
    assert isinstance(histories[2], MaxScalarHistory)
    assert histories[2].name == "num_correct_predictions"
    assert isinstance(histories[3], MinScalarHistory)
    assert histories[3].name == "num_incorrect_predictions"


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_extended_accuracy_state_get_histories_prefix_suffix(prefix: str, suffix: str):
    histories = ExtendedAccuracyState().get_histories(prefix, suffix)
    assert len(histories) == 4
    assert isinstance(histories[0], MaxScalarHistory)
    assert histories[0].name == f"{prefix}accuracy{suffix}"
    assert isinstance(histories[1], MinScalarHistory)
    assert histories[1].name == f"{prefix}error{suffix}"
    assert isinstance(histories[2], MaxScalarHistory)
    assert histories[2].name == f"{prefix}num_correct_predictions{suffix}"
    assert isinstance(histories[3], MinScalarHistory)
    assert histories[3].name == f"{prefix}num_incorrect_predictions{suffix}"


def test_extended_accuracy_state_reset():
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert state.num_predictions == 4
    state.reset()
    assert state.num_predictions == 0


def test_extended_accuracy_state_update_1d():
    state = ExtendedAccuracyState()
    state.update(torch.ones(4))
    assert state._meter.count == 4
    assert state._meter.sum() == 4.0


def test_extended_accuracy_state_update_2d():
    state = ExtendedAccuracyState()
    state.update(torch.ones(2, 3))
    assert state._meter.count == 6
    assert state._meter.sum() == 6.0


def test_extended_accuracy_state_value_correct():
    state = ExtendedAccuracyState()
    state.update(torch.ones(2, 3))
    assert state.value() == {
        "accuracy": 1.0,
        "error": 0.0,
        "num_correct_predictions": 6,
        "num_incorrect_predictions": 0,
        "num_predictions": 6,
    }


def test_extended_accuracy_state_value_partially_correct():
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert state.value() == {
        "accuracy": 0.5,
        "error": 0.5,
        "num_correct_predictions": 2,
        "num_incorrect_predictions": 2,
        "num_predictions": 4,
    }


def test_extended_accuracy_state_value_incorrect():
    state = ExtendedAccuracyState()
    state.update(torch.zeros(2, 3))
    assert state.value() == {
        "accuracy": 0.0,
        "error": 1.0,
        "num_correct_predictions": 0,
        "num_incorrect_predictions": 6,
        "num_predictions": 6,
    }


@mark.parametrize("prefix", ("", "prefix_"))
@mark.parametrize("suffix", ("", "_suffix"))
def test_extended_accuracy_state_value_prefix_suffix(prefix: str, suffix: str):
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert state.value(prefix, suffix) == {
        f"{prefix}accuracy{suffix}": 0.5,
        f"{prefix}error{suffix}": 0.5,
        f"{prefix}num_correct_predictions{suffix}": 2,
        f"{prefix}num_incorrect_predictions{suffix}": 2,
        f"{prefix}num_predictions{suffix}": 4,
    }


def test_extended_accuracy_state_value_empty():
    state = ExtendedAccuracyState()
    with raises(EmptyMetricError):
        state.value()


#################################
#     Tests for setup_state     #
#################################


def test_setup_state_object():
    state = MeanErrorState()
    assert setup_state(state) is state


def test_setup_state_dict():
    assert isinstance(
        setup_state({OBJECT_TARGET: "gravitorch.models.metrics.state.MeanErrorState"}),
        MeanErrorState,
    )
