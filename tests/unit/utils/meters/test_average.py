import math
from unittest.mock import Mock, patch

from pytest import raises

from gravitorch.distributed.ddp import SUM
from gravitorch.utils.meters import AverageMeter, EmptyMeterError

##################################
#     Tests for AverageMeter     #
##################################


def test_average_meter_repr():
    assert repr(AverageMeter()) == "AverageMeter(count=0, total=0.0)"


def test_average_meter_str():
    assert (
        str(AverageMeter(total=6, count=2))
        == "AverageMeter\n  average : 3.0\n  count   : 2\n  total   : 6.0"
    )


def test_average_meter_str_empty():
    assert str(AverageMeter()) == (
        "AverageMeter\n  average : N/A (empty)\n  count   : 0\n  total   : 0.0"
    )


def test_average_meter_all_reduce():
    meter = AverageMeter(total=122.0, count=10)
    meter_reduced = meter.all_reduce()
    assert meter.equal(AverageMeter(total=122.0, count=10))
    assert meter_reduced.equal(AverageMeter(total=122.0, count=10))


def test_average_meter_all_reduce_empty():
    assert AverageMeter().all_reduce().equal(AverageMeter())


def test_average_meter_all_reduce_total_reduce():
    meter = AverageMeter(total=122.0, count=10)
    reduce_mock = Mock(side_effect=lambda variable, op: variable + 1)
    with patch("gravitorch.utils.meters.average.sync_reduce", reduce_mock):
        meter_reduced = meter.all_reduce()
        assert meter.equal(AverageMeter(total=122.0, count=10))
        assert meter_reduced.equal(AverageMeter(total=123.0, count=11))
        assert reduce_mock.call_args_list == [((122.0, SUM), {}), ((10, SUM), {})]


def test_average_meter_average():
    assert AverageMeter(total=6, count=2).average() == 3.0


def test_average_meter_average_empty():
    meter = AverageMeter()
    with raises(EmptyMeterError):
        meter.average()


def test_average_meter_clone():
    meter = AverageMeter(total=122.0, count=10)
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(AverageMeter(total=122.0, count=10))


def test_average_meter_clone_empty():
    meter = AverageMeter()
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(AverageMeter())


def test_average_meter_equal_true():
    assert AverageMeter(total=122.0, count=10).equal(AverageMeter(total=122.0, count=10))


def test_average_meter_equal_true_empty():
    assert AverageMeter().equal(AverageMeter())


def test_average_meter_equal_false_different_total():
    assert not AverageMeter(total=121.0, count=10).equal(AverageMeter(total=122.0, count=10))


def test_average_meter_equal_false_different_count():
    assert not AverageMeter(total=122.0, count=10).equal(AverageMeter(total=122.0, count=9))


def test_average_meter_equal_false_different_type():
    assert not AverageMeter(total=122.0, count=10).equal(1)


def test_average_meter_merge():
    meter = AverageMeter(total=122.0, count=10)
    meter_merged = meter.merge(
        [AverageMeter(total=1.0, count=4), AverageMeter(), AverageMeter(total=-2.0, count=2)]
    )
    assert meter.equal(AverageMeter(total=122.0, count=10))
    assert meter_merged.equal(AverageMeter(total=121.0, count=16))


def test_average_meter_merge_():
    meter = AverageMeter(total=122.0, count=10)
    meter.merge_(
        [AverageMeter(total=1.0, count=4), AverageMeter(), AverageMeter(total=-2.0, count=2)]
    )
    assert meter.equal(AverageMeter(total=121.0, count=16))


def test_average_meter_load_state_dict():
    meter = AverageMeter()
    meter.load_state_dict({"count": 10, "total": 122.0})
    assert meter.equal(AverageMeter(total=122.0, count=10))


def test_average_meter_reset():
    meter = AverageMeter(total=122.0, count=10)
    meter.reset()
    assert meter.equal(AverageMeter())


def test_average_meter_reset_empty():
    meter = AverageMeter()
    meter.reset()
    assert meter.equal(AverageMeter())


def test_average_meter_state_dict():
    assert AverageMeter(total=19.0, count=4).state_dict() == {"count": 4, "total": 19}


def test_average_meter_state_dict_empty():
    assert AverageMeter().state_dict() == {"count": 0, "total": 0}


def test_average_meter_sum():
    assert AverageMeter(total=6, count=2).sum() == 6.0


def test_average_meter_sum_empty():
    meter = AverageMeter()
    with raises(EmptyMeterError):
        meter.sum()


def test_average_meter_update_4():
    meter = AverageMeter()
    meter.update(4)
    assert meter.equal(AverageMeter(total=4.0, count=1))


def test_average_meter_update_4_and_2():
    meter = AverageMeter()
    meter.update(4)
    meter.update(2)
    assert meter.equal(AverageMeter(total=6.0, count=2))


def test_average_meter_update_with_num_examples():
    meter = AverageMeter()
    meter.update(4, num_examples=2)
    meter.update(2)
    meter.update(2)
    assert meter.equal(AverageMeter(total=12.0, count=4))


def test_average_meter_update_nan():
    meter = AverageMeter()
    meter.update(float("NaN"))
    assert math.isnan(meter.total)
    assert meter.count == 1


def test_average_meter_update_inf():
    meter = AverageMeter()
    meter.update(float("inf"))
    assert meter.equal(AverageMeter(total=float("inf"), count=1))
