import math
from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from pytest import mark, raises

from gravitorch.distributed.ddp import MAX, MIN, SUM
from gravitorch.utils.meters import (
    EmptyMeterError,
    ExtremaTensorMeter,
    MeanTensorMeter,
    TensorMeter,
    TensorMeter2,
)

#####################################
#     Tests for MeanTensorMeter     #
#####################################


def test_mean_tensor_meter_repr():
    assert repr(MeanTensorMeter(count=8, total=20.0)) == "MeanTensorMeter(count=8, total=20.0)"


def test_mean_tensor_meter_str():
    assert str(MeanTensorMeter(count=8, total=20.0)) == "MeanTensorMeter(count=8, total=20.0)"


def test_mean_tensor_meter_str_empty():
    assert str(MeanTensorMeter()) == "MeanTensorMeter(count=0, total=0)"


def test_mean_tensor_meter_count():
    assert MeanTensorMeter(count=8).count == 8


def test_mean_tensor_meter_count_empty():
    assert MeanTensorMeter().count == 0


def test_mean_tensor_meter_total():
    assert MeanTensorMeter(total=12.0).total == 12.0


def test_mean_tensor_meter_total_empty():
    assert MeanTensorMeter().total == 0


def test_mean_tensor_meter_reset():
    meter = MeanTensorMeter(count=8, total=20.0)
    meter.reset()
    assert meter.equal(MeanTensorMeter())


def test_mean_tensor_meter_update():
    meter = MeanTensorMeter()
    meter.update(torch.arange(6))
    meter.update(torch.tensor([4.0, 1.0]))
    assert meter.equal(MeanTensorMeter(count=8, total=20.0))


def test_mean_tensor_meter_update_1d():
    meter = MeanTensorMeter()
    meter.update(torch.arange(6))
    assert meter.equal(MeanTensorMeter(count=6, total=15))


def test_mean_tensor_meter_update_2d():
    meter = MeanTensorMeter()
    meter.update(torch.arange(6).view(2, 3))
    assert meter.equal(MeanTensorMeter(count=6, total=15))


def test_mean_tensor_meter_update_3d():
    meter = MeanTensorMeter()
    meter.update(torch.ones(2, 3, 4))
    assert meter.equal(MeanTensorMeter(count=24, total=24.0))


def test_mean_tensor_meter_update_float():
    meter = MeanTensorMeter()
    meter.update(torch.tensor([4.0, 1.0], dtype=torch.float))
    assert meter.equal(MeanTensorMeter(count=2, total=5.0))


def test_mean_tensor_meter_update_long():
    meter = MeanTensorMeter()
    meter.update(torch.tensor([4, 1], dtype=torch.long))
    assert meter.equal(MeanTensorMeter(count=2, total=5))


def test_mean_tensor_meter_update_nan():
    meter = MeanTensorMeter()
    meter.update(torch.tensor(float("NaN")))
    assert math.isnan(meter.sum())
    assert meter.count == 1


def test_mean_tensor_meter_update_inf():
    meter = MeanTensorMeter()
    meter.update(torch.tensor(float("inf")))
    assert meter.equal(MeanTensorMeter(count=1, total=float("inf")))


def test_mean_tensor_meter_average():
    assert MeanTensorMeter(count=8, total=20.0).average() == 2.5


def test_mean_tensor_meter_average_empty():
    meter = MeanTensorMeter()
    with raises(EmptyMeterError):
        meter.average()


def test_mean_tensor_meter_mean():
    assert MeanTensorMeter(count=8, total=20.0).mean() == 2.5


def test_mean_tensor_meter_mean_empty():
    meter = MeanTensorMeter()
    with raises(EmptyMeterError):
        meter.mean()


def test_mean_tensor_meter_sum_int():
    total = MeanTensorMeter(count=8, total=22).sum()
    assert total == 22
    assert isinstance(total, int)


def test_mean_tensor_meter_sum_float():
    total = MeanTensorMeter(count=8, total=20.0).sum()
    assert total == 20.0
    assert isinstance(total, float)


def test_mean_tensor_meter_sum_empty():
    meter = MeanTensorMeter()
    with raises(EmptyMeterError):
        meter.sum()


def test_mean_tensor_meter_all_reduce():
    meter = MeanTensorMeter(count=10, total=122.0)
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(MeanTensorMeter(count=10, total=122.0))
    assert meter_reduced.equal(MeanTensorMeter(count=10, total=122.0))


def test_mean_tensor_meter_all_reduce_empty():
    meter = MeanTensorMeter()
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(MeanTensorMeter())
    assert meter_reduced.equal(MeanTensorMeter())


def test_mean_tensor_meter_all_reduce_sum_reduce():
    meter = MeanTensorMeter(count=10, total=122.0)
    reduce_mock = Mock(side_effect=lambda variable, op: variable + 1)
    with patch("gravitorch.utils.meters.tensor.sync_reduce", reduce_mock):
        meter_reduced = meter.all_reduce()
        assert meter.equal(MeanTensorMeter(count=10, total=122.0))
        assert meter_reduced.equal(MeanTensorMeter(count=11, total=123.0))
        assert reduce_mock.call_args_list == [((10, SUM), {}), ((122.0, SUM), {})]


def test_mean_tensor_meter_clone():
    meter = MeanTensorMeter(count=10, total=122.0)
    meter_cloned = meter.clone()
    assert meter_cloned is not meter
    assert meter.equal(MeanTensorMeter(count=10, total=122.0))
    assert meter_cloned.equal(MeanTensorMeter(count=10, total=122.0))


def test_mean_tensor_meter_clone_empty():
    meter = MeanTensorMeter()
    meter_cloned = meter.clone()
    assert meter_cloned is not meter
    assert meter.equal(MeanTensorMeter())
    assert meter_cloned.equal(MeanTensorMeter())


def test_mean_tensor_meter_equal_true():
    assert MeanTensorMeter(total=122.0, count=10).equal(MeanTensorMeter(total=122.0, count=10))


def test_mean_tensor_meter_equal_true_empty():
    assert MeanTensorMeter().equal(MeanTensorMeter())


def test_mean_tensor_meter_equal_false_different_count():
    assert not MeanTensorMeter(total=122.0, count=10).equal(MeanTensorMeter(total=122.0, count=9))


def test_mean_tensor_meter_equal_false_different_total():
    assert not MeanTensorMeter(total=122.0, count=10).equal(MeanTensorMeter(total=12.0, count=10))


def test_mean_tensor_meter_equal_false_different_type():
    assert not MeanTensorMeter(total=122.0, count=10).equal(1)


def test_mean_tensor_meter_merge():
    meter = MeanTensorMeter(total=122.0, count=10)
    meter_merged = meter.merge(
        [
            MeanTensorMeter(total=1.0, count=4),
            MeanTensorMeter(),
            MeanTensorMeter(total=-2.0, count=2),
        ]
    )
    assert meter.equal(MeanTensorMeter(total=122.0, count=10))
    assert meter_merged.equal(MeanTensorMeter(total=121.0, count=16))


def test_mean_tensor_meter_merge_():
    meter = MeanTensorMeter(total=122.0, count=10)
    meter.merge_(
        [
            MeanTensorMeter(total=1.0, count=4),
            MeanTensorMeter(),
            MeanTensorMeter(total=-2.0, count=2),
        ]
    )
    assert meter.equal(MeanTensorMeter(total=121.0, count=16))


def test_mean_tensor_meter_load_state_dict():
    meter = MeanTensorMeter()
    meter.load_state_dict({"count": 10, "total": 122.0})
    assert meter.equal(MeanTensorMeter(count=10, total=122.0))


def test_mean_tensor_meter_state_dict():
    assert MeanTensorMeter(count=6, total=15.0).state_dict() == {"count": 6, "total": 15.0}


def test_mean_tensor_meter_state_dict_empty():
    assert MeanTensorMeter().state_dict() == {
        "count": 0,
        "total": 0,
    }


########################################
#     Tests for ExtremaTensorMeter     #
########################################


def test_extrema_tensor_meter_repr():
    assert (
        repr(ExtremaTensorMeter(count=8, min_value=-2.0, max_value=5.0))
        == "ExtremaTensorMeter(count=8, min_value=-2.0, max_value=5.0)"
    )


def test_extrema_tensor_meter_str():
    assert (
        str(ExtremaTensorMeter(count=8, min_value=-2.0, max_value=5.0))
        == "ExtremaTensorMeter(count=8, min_value=-2.0, max_value=5.0)"
    )


def test_extrema_tensor_meter_str_empty():
    assert str(ExtremaTensorMeter()) == "ExtremaTensorMeter(count=0, min_value=inf, max_value=-inf)"


def test_extrema_tensor_meter_count():
    assert ExtremaTensorMeter(count=8).count == 8


def test_extrema_tensor_meter_count_empty():
    assert ExtremaTensorMeter().count == 0


def test_extrema_tensor_meter_reset():
    meter = ExtremaTensorMeter(count=8, min_value=-2.0, max_value=5.0)
    meter.reset()
    assert meter.equal(ExtremaTensorMeter())


def test_extrema_tensor_meter_update():
    meter = ExtremaTensorMeter()
    meter.update(torch.arange(6, dtype=torch.float))
    meter.update(torch.tensor([4.0, -1.0]))
    assert meter.equal(ExtremaTensorMeter(count=8, min_value=-1.0, max_value=5.0))


def test_extrema_tensor_meter_update_1d():
    meter = ExtremaTensorMeter()
    meter.update(torch.arange(6, dtype=torch.float))
    assert meter.equal(ExtremaTensorMeter(count=6, min_value=0.0, max_value=5.0))


def test_extrema_tensor_meter_update_2d():
    meter = ExtremaTensorMeter()
    meter.update(torch.arange(6, dtype=torch.float).view(2, 3))
    assert meter.equal(ExtremaTensorMeter(count=6, min_value=0.0, max_value=5.0))


def test_extrema_tensor_meter_update_3d():
    meter = ExtremaTensorMeter()
    meter.update(torch.ones(2, 3, 4))
    assert meter.equal(ExtremaTensorMeter(count=24, min_value=1.0, max_value=1.0))


def test_extrema_tensor_meter_update_nan():
    meter = ExtremaTensorMeter()
    meter.update(torch.tensor(float("NaN")))
    assert meter.equal(ExtremaTensorMeter(count=1, min_value=float("inf"), max_value=float("-inf")))


def test_extrema_tensor_meter_update_inf():
    meter = ExtremaTensorMeter()
    meter.update(torch.tensor(float("inf")))
    assert meter.equal(ExtremaTensorMeter(count=1, min_value=float("inf"), max_value=float("inf")))


@mark.parametrize("max_value", (0.0, 5.0))
def test_extrema_tensor_meter_max(max_value: float):
    assert ExtremaTensorMeter(count=8, min_value=0.0, max_value=max_value).max() == max_value


def test_extrema_tensor_meter_max_empty():
    meter = ExtremaTensorMeter()
    with raises(EmptyMeterError):
        meter.max()


@mark.parametrize("min_value", (0.0, -5.0))
def test_extrema_tensor_meter_min(min_value: float):
    assert ExtremaTensorMeter(count=8, min_value=min_value, max_value=5.0).min() == min_value


def test_extrema_tensor_meter_min_empty():
    meter = ExtremaTensorMeter()
    with raises(EmptyMeterError):
        meter.min()


def test_extrema_tensor_meter_all_reduce():
    meter = ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0)
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0))
    assert meter_reduced.equal(ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0))


def test_extrema_tensor_meter_all_reduce_empty():
    meter = ExtremaTensorMeter()
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(ExtremaTensorMeter())
    assert meter_reduced.equal(ExtremaTensorMeter())


def test_extrema_tensor_meter_all_reduce_ops():
    meter = ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0)
    reduce_mock = Mock(side_effect=lambda variable, op: variable + 1)
    with patch("gravitorch.utils.meters.tensor.sync_reduce", reduce_mock):
        meter_reduced = meter.all_reduce()
        assert meter.equal(ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0))
        assert meter_reduced.equal(ExtremaTensorMeter(count=7, min_value=-1.0, max_value=6.0))
        assert reduce_mock.call_args_list == [
            ((6, SUM), {}),
            ((-2.0, MIN), {}),
            ((5.0, MAX), {}),
        ]


def test_extrema_tensor_meter_clone():
    meter = ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0)
    meter_cloned = meter.clone()
    assert meter_cloned is not meter
    assert meter.equal(ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0))
    assert meter_cloned.equal(ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0))


def test_extrema_tensor_meter_clone_empty():
    meter = ExtremaTensorMeter()
    meter_cloned = meter.clone()
    assert meter_cloned is not meter
    assert meter.equal(ExtremaTensorMeter())
    assert meter_cloned.equal(ExtremaTensorMeter())


def test_extrema_tensor_meter_equal_true():
    assert ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0).equal(
        ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0)
    )


def test_extrema_tensor_meter_equal_true_empty():
    assert ExtremaTensorMeter().equal(ExtremaTensorMeter())


def test_extrema_tensor_meter_equal_false_different_count():
    assert not ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0).equal(
        ExtremaTensorMeter(count=5, min_value=-2.0, max_value=5.0)
    )


def test_extrema_tensor_meter_equal_false_different_min_value():
    assert not ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0).equal(
        ExtremaTensorMeter(count=6, min_value=-3.0, max_value=5.0)
    )


def test_extrema_tensor_meter_equal_false_different_max_value():
    assert not ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0).equal(
        ExtremaTensorMeter(count=6, min_value=-2.0, max_value=6.0)
    )


def test_extrema_tensor_meter_equal_false_different_type():
    assert not ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0).equal(1)


def test_extrema_tensor_meter_merge():
    meter = ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0)
    meter_merged = meter.merge(
        [
            ExtremaTensorMeter(count=4, min_value=-3.0, max_value=2.0),
            ExtremaTensorMeter(),
            ExtremaTensorMeter(count=2, min_value=-1.0, max_value=7.0),
        ]
    )
    assert meter.equal(ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0))
    assert meter_merged.equal(ExtremaTensorMeter(count=12, min_value=-3.0, max_value=7.0))


def test_extrema_tensor_meter_merge_():
    meter = ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0)
    meter.merge_(
        [
            ExtremaTensorMeter(count=4, min_value=-3.0, max_value=2.0),
            ExtremaTensorMeter(),
            ExtremaTensorMeter(count=2, min_value=-1.0, max_value=7.0),
        ]
    )
    assert meter.equal(ExtremaTensorMeter(count=12, min_value=-3.0, max_value=7.0))


def test_extrema_tensor_meter_load_state_dict():
    meter = ExtremaTensorMeter()
    meter.load_state_dict({"count": 6, "min_value": -2.0, "max_value": 5.0})
    assert meter.min() == -2.0
    assert meter.max() == 5.0
    assert meter.count == 6


def test_extrema_tensor_meter_state_dict():
    assert ExtremaTensorMeter(count=6, min_value=-2.0, max_value=5.0).state_dict() == {
        "count": 6,
        "min_value": -2.0,
        "max_value": 5.0,
    }


def test_extrema_tensor_meter_state_dict_empty():
    assert ExtremaTensorMeter().state_dict() == {
        "count": 0,
        "min_value": float("inf"),
        "max_value": float("-inf"),
    }


#################################
#     Tests for TensorMeter     #
#################################


def test_tensor_meter_repr():
    assert repr(TensorMeter()) == "TensorMeter(count=0, total=0.0, min_value=inf, max_value=-inf)"


def test_tensor_meter_str():
    assert str(TensorMeter(count=8, total=20.0, min_value=0.0, max_value=5.0)) == (
        "TensorMeter\n"
        "  count   : 8\n"
        "  sum     : 20.0\n"
        "  average : 2.5\n"
        "  min     : 0.0\n"
        "  max     : 5.0"
    )


def test_tensor_meter_str_empty():
    meter = TensorMeter()
    assert str(meter) == (
        "TensorMeter\n"
        "  count   : 0\n"
        "  sum     : N/A (empty)\n"
        "  average : N/A (empty)\n"
        "  min     : N/A (empty)\n"
        "  max     : N/A (empty)"
    )


def test_tensor_meter_count():
    assert TensorMeter(count=8).count == 8


def test_tensor_meter_count_empty():
    assert TensorMeter().count == 0


def test_tensor_meter_total():
    assert TensorMeter(total=12.0).total == 12.0


def test_tensor_meter_total_empty():
    assert TensorMeter().total == 0


def test_tensor_meter_all_reduce():
    meter = TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0)
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0))
    assert meter_reduced.equal(TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0))


def test_tensor_meter_all_reduce_empty():
    meter = TensorMeter()
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(TensorMeter())
    assert meter_reduced.equal(TensorMeter())


def test_tensor_meter_all_reduce_sum_reduce():
    meter = TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0)
    reduce_mock = Mock(side_effect=lambda variable, op: variable + 1)
    with patch("gravitorch.utils.meters.tensor.sync_reduce", reduce_mock):
        meter_reduced = meter.all_reduce()
        assert meter.equal(TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0))
        assert meter_reduced.equal(
            TensorMeter(count=11, total=123.0, min_value=-4.0, max_value=21.0)
        )
        assert reduce_mock.call_args_list == [
            ((10, SUM), {}),
            ((122.0, SUM), {}),
            ((-5.0, MIN), {}),
            ((20.0, MAX), {}),
        ]


def test_tensor_meter_average():
    assert TensorMeter(count=8, total=20.0, min_value=0.0, max_value=5.0).average() == 2.5


def test_tensor_meter_average_empty():
    meter = TensorMeter()
    with raises(EmptyMeterError):
        meter.average()


@mark.parametrize("max_value", (0.0, 5.0))
def test_tensor_meter_max(max_value: float):
    assert TensorMeter(count=8, total=20.0, min_value=0.0, max_value=max_value).max() == max_value


def test_tensor_meter_max_empty():
    meter = TensorMeter()
    with raises(EmptyMeterError):
        meter.max()


def test_tensor_meter_mean():
    assert TensorMeter(count=8, total=20.0, min_value=0.0, max_value=5.0).mean() == 2.5


def test_tensor_meter_mean_empty():
    meter = TensorMeter()
    with raises(EmptyMeterError):
        meter.mean()


@mark.parametrize("min_value", (0.0, -5.0))
def test_tensor_meter_min(min_value: float):
    assert TensorMeter(count=8, total=20.0, min_value=min_value, max_value=5.0).min() == min_value


def test_tensor_meter_min_empty():
    meter = TensorMeter()
    with raises(EmptyMeterError):
        meter.min()


def test_tensor_meter_clone():
    meter = TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0)
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter.equal(TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0))
    assert meter_cloned.equal(TensorMeter(count=10, total=122.0, min_value=-5.0, max_value=20.0))


def test_tensor_meter_clone_empty():
    meter = TensorMeter()
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter.equal(TensorMeter())
    assert meter_cloned.equal(TensorMeter())


def test_tensor_meter_equal_true():
    assert TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=5.0).equal(
        TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=5.0)
    )


def test_tensor_meter_equal_true_empty():
    assert TensorMeter().equal(TensorMeter())


def test_tensor_meter_equal_false_different_count():
    assert not TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=5.0).equal(
        TensorMeter(count=5, total=122.0, min_value=-2.0, max_value=5.0)
    )


def test_tensor_meter_equal_false_different_total():
    assert not TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=5.0).equal(
        TensorMeter(count=6, total=121.0, min_value=-2.0, max_value=5.0)
    )


def test_tensor_meter_equal_false_different_min_value():
    assert not TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=5.0).equal(
        TensorMeter(count=6, total=122.0, min_value=-3.0, max_value=5.0)
    )


def test_tensor_meter_equal_false_different_max_value():
    assert not TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=5.0).equal(
        TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=6.0)
    )


def test_tensor_meter_equal_false_different_type():
    assert not TensorMeter(count=6, total=122.0, min_value=-2.0, max_value=5.0).equal(1)


def test_tensor_meter_merge():
    meter = TensorMeter(count=10, total=122.0, min_value=-2.0, max_value=5.0)
    meter_merged = meter.merge(
        [
            TensorMeter(count=5, total=10.0, min_value=-1.0, max_value=6.0),
            TensorMeter(),
            TensorMeter(count=2, total=-5.0, min_value=-3.0, max_value=2.0),
        ]
    )
    assert meter.equal(TensorMeter(count=10, total=122.0, min_value=-2.0, max_value=5.0))
    assert meter_merged.equal(TensorMeter(count=17, total=127.0, min_value=-3.0, max_value=6.0))


def test_tensor_meter_merge_():
    meter = TensorMeter(count=10, total=122.0, min_value=-2.0, max_value=5.0)
    meter.merge_(
        [
            TensorMeter(count=5, total=10.0, min_value=-1.0, max_value=6.0),
            TensorMeter(),
            TensorMeter(count=2, total=-5.0, min_value=-3.0, max_value=2.0),
        ]
    )
    assert meter.equal(TensorMeter(count=17, total=127.0, min_value=-3.0, max_value=6.0))


def test_tensor_meter_load_state_dict():
    meter = TensorMeter()
    meter.load_state_dict({"count": 10, "max_value": 5.0, "min_value": -2.0, "total": 122.0})
    assert meter.equal(TensorMeter(count=10, total=122.0, min_value=-2.0, max_value=5.0))


def test_tensor_meter_state_dict():
    assert TensorMeter(count=6, total=15.0, min_value=0.0, max_value=5.0).state_dict() == {
        "count": 6,
        "max_value": 5.0,
        "min_value": 0.0,
        "total": 15.0,
    }


def test_tensor_meter_state_dict_empty():
    assert TensorMeter().state_dict() == {
        "count": 0,
        "max_value": float("-inf"),
        "min_value": float("inf"),
        "total": 0.0,
    }


@mark.parametrize("total", (15.0, 20.0))
def test_tensor_meter_sum(total: float):
    assert TensorMeter(count=6, total=total, min_value=0.0, max_value=5.0).sum() == total


def test_tensor_meter_sum_empty():
    meter = TensorMeter()
    with raises(EmptyMeterError):
        meter.sum()


def test_tensor_meter_update():
    meter = TensorMeter()
    meter.update(torch.arange(6))
    meter.update(torch.tensor([4.0, 1.0]))
    assert meter.equal(TensorMeter(count=8, total=20.0, min_value=0.0, max_value=5.0))


def test_tensor_meter_update_nan():
    meter = TensorMeter()
    meter.update(torch.tensor(float("NaN")))
    assert meter.max() == float("-inf")
    assert meter.min() == float("inf")
    assert math.isnan(meter.sum())
    assert meter.count == 1


def test_tensor_meter_update_inf():
    meter = TensorMeter()
    meter.update(torch.tensor(float("inf")))
    assert meter.equal(
        TensorMeter(count=1, total=float("inf"), min_value=float("inf"), max_value=float("inf"))
    )


##################################
#     Tests for TensorMeter2     #
##################################


def test_tensor_meter2_repr():
    assert repr(TensorMeter2(torch.arange(6))) == "TensorMeter2(count=6)"


def test_tensor_meter2_str():
    assert repr(TensorMeter2(torch.arange(6))) == "TensorMeter2(count=6)"


def test_tensor_meter2_str_empty():
    assert repr(TensorMeter2()) == "TensorMeter2(count=0)"


def test_tensor_meter2_count():
    assert TensorMeter2(torch.arange(6)).count == 6


def test_tensor_meter2_count_empty():
    assert TensorMeter2().count == 0


def test_tensor_meter2_reset():
    meter = TensorMeter2(torch.arange(6))
    meter.reset()
    assert meter.equal(TensorMeter2())


def test_tensor_meter2_update():
    meter = TensorMeter2()
    meter.update(torch.arange(6))
    meter.update(torch.tensor([4.0, 1.0]))
    assert meter.equal(
        TensorMeter2(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 1.0], dtype=torch.float))
    )


def test_tensor_meter2_update_1d():
    meter = TensorMeter2()
    meter.update(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float))
    assert meter.equal(
        TensorMeter2(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float))
    )


def test_tensor_meter2_update_2d():
    meter = TensorMeter2()
    meter.update(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float))
    assert meter.equal(
        TensorMeter2(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float))
    )


def test_tensor_meter2_update_3d():
    meter = TensorMeter2()
    meter.update(torch.ones(2, 3, 4))
    assert meter.equal(TensorMeter2(torch.ones(24)))


def test_tensor_meter2_update_float():
    meter = TensorMeter2()
    meter.update(torch.tensor([4.0, 1.0], dtype=torch.float))
    assert meter.equal(TensorMeter2(torch.tensor([4.0, 1.0], dtype=torch.float)))


def test_tensor_meter2_update_long():
    meter = TensorMeter2()
    meter.update(torch.tensor([4, 1], dtype=torch.long))
    assert meter.equal(TensorMeter2(torch.tensor([4, 1], dtype=torch.long)))


def test_tensor_meter2_update_nan():
    meter = TensorMeter2()
    meter.update(torch.tensor(float("NaN")))
    assert math.isnan(meter.sum())
    assert meter.count == 1


def test_tensor_meter2_update_inf():
    meter = TensorMeter2()
    meter.update(torch.tensor(float("inf")))
    assert meter.equal(TensorMeter2(torch.tensor([float("inf")])))


def test_tensor_meter2_average_float():
    assert TensorMeter2(torch.tensor([-2.0, 1.0, 7.0], dtype=torch.float)).average() == 2.0


def test_tensor_meter2_average_long():
    assert TensorMeter2(torch.tensor([-2, 1, 7], dtype=torch.long)).average() == 2.0


def test_tensor_meter2_average_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.average()


def test_tensor_meter2_max_float():
    max_value = TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).max()
    assert max_value == 7.0
    assert isinstance(max_value, float)


def test_tensor_meter2_max_long():
    max_value = TensorMeter2(torch.tensor([-3, 1, 7], dtype=torch.long)).max()
    assert max_value == 7
    assert isinstance(max_value, int)


def test_tensor_meter2_max_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.max()


def test_tensor_meter2_mean_float():
    assert TensorMeter2(torch.tensor([-2.0, 1.0, 7.0], dtype=torch.float)).mean() == 2.0


def test_tensor_meter2_mean_long():
    assert TensorMeter2(torch.tensor([-2, 1, 7], dtype=torch.long)).mean() == 2.0


def test_tensor_meter2_mean_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.mean()


def test_tensor_meter2_median_float():
    assert TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).median() == 1.0


def test_tensor_meter2_median_long():
    assert TensorMeter2(torch.tensor([-3, 1, 7], dtype=torch.long)).median() == 1


def test_tensor_meter2_median_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.median()


def test_tensor_meter2_min_float():
    min_value = TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).min()
    assert min_value == -3.0
    assert isinstance(min_value, float)


def test_tensor_meter2_min_long():
    min_value = TensorMeter2(torch.tensor([-3, 1, 7], dtype=torch.long)).min()
    assert min_value == -3
    assert isinstance(min_value, int)


def test_tensor_meter2_min_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.min()


def test_tensor_meter2_quantile_float():
    assert (
        TensorMeter2(torch.arange(11, dtype=torch.float))
        .quantile(q=torch.tensor([0.5, 0.9], dtype=torch.float))
        .equal(torch.tensor([5.0, 9.0], dtype=torch.float))
    )


def test_tensor_meter2_quantile_long():
    assert (
        TensorMeter2(torch.arange(11, dtype=torch.long))
        .quantile(q=torch.tensor([0.5, 0.9], dtype=torch.float))
        .equal(torch.tensor([5.0, 9.0], dtype=torch.float))
    )


def test_tensor_meter2_quantile_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.quantile(q=torch.tensor([0.5, 0.9]))


def test_tensor_meter2_std_float():
    assert TensorMeter2(torch.ones(3, dtype=torch.float)).std() == 0.0


def test_tensor_meter2_std_long():
    assert TensorMeter2(torch.ones(3, dtype=torch.long)).std() == 0.0


def test_tensor_meter2_std_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.std()


def test_tensor_meter2_sum_float():
    total = TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).sum()
    assert total == 5.0
    assert isinstance(total, float)


def test_tensor_meter2_sum_long():
    total = TensorMeter2(torch.arange(6)).sum()
    assert total == 15
    assert isinstance(total, int)


def test_tensor_meter2_sum_empty():
    meter = TensorMeter2()
    with raises(EmptyMeterError):
        meter.sum()


def test_tensor_meter2_all_reduce():
    meter = TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)))
    assert meter_reduced.equal(TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)))


def test_tensor_meter2_all_reduce_empty():
    meter = TensorMeter2()
    meter_reduced = meter.all_reduce()
    assert meter_reduced is not meter
    assert meter.equal(TensorMeter2())
    assert meter_reduced.equal(TensorMeter2())


def test_tensor_meter2_clone():
    meter = TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    meter_cloned = meter.clone()
    assert meter_cloned is not meter
    assert meter.equal(TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)))
    assert meter_cloned.equal(TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)))


def test_tensor_meter2_clone_empty():
    meter = TensorMeter2()
    meter_cloned = meter.clone()
    assert meter_cloned is not meter
    assert meter.equal(TensorMeter2())
    assert meter_cloned.equal(TensorMeter2())


def test_tensor_meter2_equal_true():
    assert TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).equal(
        TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    )


def test_tensor_meter2_equal_true_empty():
    assert TensorMeter2().equal(TensorMeter2())


def test_tensor_meter2_equal_false_different_values():
    assert not TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).equal(
        TensorMeter2(torch.tensor([-3.0, 1.0, 7.0, 2.0], dtype=torch.float))
    )


def test_tensor_meter2_equal_false_different_type():
    assert not TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).equal(1)


def test_tensor_meter2_merge():
    meter = TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    meter_merged = meter.merge(
        [
            TensorMeter2(torch.tensor([2.0, 5.0], dtype=torch.float)),
            TensorMeter2(),
            TensorMeter2(torch.tensor([-1.0], dtype=torch.float)),
        ]
    )
    assert meter.equal(TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)))
    assert meter_merged.equal(
        TensorMeter2(torch.tensor([-3.0, 1.0, 7.0, 2.0, 5.0, -1.0], dtype=torch.float))
    )


def test_tensor_meter2_merge_():
    meter = TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    meter.merge_(
        [
            TensorMeter2(torch.tensor([2.0, 5.0], dtype=torch.float)),
            TensorMeter2(),
            TensorMeter2(torch.tensor([-1.0], dtype=torch.float)),
        ]
    )
    assert meter.equal(
        TensorMeter2(torch.tensor([-3.0, 1.0, 7.0, 2.0, 5.0, -1.0], dtype=torch.float))
    )


def test_tensor_meter2_load_state_dict():
    meter = TensorMeter2()
    meter.load_state_dict({"values": torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)})
    assert meter.equal(TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)))


def test_tensor_meter2_state_dict():
    assert objects_are_equal(
        TensorMeter2(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)).state_dict(),
        {"values": torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float)},
    )


def test_tensor_meter2_state_dict_empty():
    assert objects_are_equal(TensorMeter2().state_dict(), {"values": torch.tensor([])})
