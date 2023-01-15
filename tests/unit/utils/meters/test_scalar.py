import math
from typing import Union
from unittest.mock import patch

from pytest import mark, raises

from gravitorch.utils.meters import EmptyMeterError, ScalarMeter

#################################
#     Tests for ScalarMeter     #
#################################


def test_scalar_meter_repr():
    assert (
        repr(ScalarMeter())
        == "ScalarMeter(count=0, total=0.0, min_value=inf, max_value=-inf, max_size=100)"
    )


@patch("gravitorch.utils.meters.ScalarMeter.std", lambda *args: 1.5)
def test_scalar_meter_str():
    assert str(
        ScalarMeter(total=6.0, count=2, max_value=4.0, min_value=2.0, values=(4.0, 2.0))
    ) == (
        "ScalarMeter\n"
        "  average : 3.0\n"
        "  count   : 2\n"
        "  max     : 4.0\n"
        "  median  : 2.0\n"
        "  min     : 2.0\n"
        "  std     : 1.5\n"
        "  sum     : 6.0"
    )


def test_scalar_meter_str_empty():
    assert str(ScalarMeter()) == (
        "ScalarMeter\n"
        "  average : N/A (empty)\n"
        "  count   : 0\n"
        "  max     : N/A (empty)\n"
        "  median  : N/A (empty)\n"
        "  min     : N/A (empty)\n"
        "  std     : N/A (empty)\n"
        "  sum     : N/A (empty)"
    )


def test_scalar_meter_count():
    assert (
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).count
        == 10
    )


def test_scalar_meter_count_empty():
    assert ScalarMeter().count == 0


def test_scalar_meter_total():
    assert (
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).total
        == 122.0
    )


def test_scalar_meter_total_empty():
    assert ScalarMeter().total == 0


def test_scalar_meter_values():
    assert ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).values == (1.0, 3.0, 5.0, 4.0, 2.0)


def test_scalar_meter_values_empty():
    assert ScalarMeter().values == tuple()


def test_scalar_meter_average():
    assert (
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).average()
        == 12.2
    )


def test_scalar_meter_average_empty():
    meter = ScalarMeter()
    with raises(EmptyMeterError):
        meter.average()


def test_scalar_meter_equal_true():
    assert ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_equal_true_empty():
    assert ScalarMeter().equal(ScalarMeter())


def test_scalar_meter_equal_false_different_count():
    assert not ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarMeter(
            total=122.0, count=9, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_equal_false_different_total():
    assert not ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarMeter(
            total=12.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_equal_false_different_max_value():
    assert not ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarMeter(
            total=122.0, count=10, max_value=16.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_equal_false_different_min_value():
    assert not ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-12.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_equal_false_different_values():
    assert not ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarMeter(total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1, 2, 3, 4, 6))
    )


def test_scalar_meter_equal_false_different_type():
    assert not ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(1)


def test_scalar_meter_load_state_dict():
    meter = ScalarMeter()
    meter.load_state_dict(
        {
            "count": 10,
            "total": 122.0,
            "values": (1.0, 3.0, 5.0, 4.0, 2.0),
            "max_value": 6.0,
            "min_value": -2.0,
        }
    )
    assert meter.equal(
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_max():
    assert (
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).max()
        == 6.0
    )


def test_scalar_meter_max_empty():
    meter = ScalarMeter()
    with raises(EmptyMeterError):
        meter.max()


def test_scalar_meter_median():
    assert (
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).median()
        == 3.0
    )


def test_scalar_meter_median_empty():
    meter = ScalarMeter()
    with raises(EmptyMeterError):
        meter.median()


def test_scalar_meter_merge():
    meter = ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    meter_merged = meter.merge(
        [
            ScalarMeter(total=12.0, count=5, max_value=8.0, min_value=-1.0, values=(1.0, 3.0)),
            ScalarMeter(),
            ScalarMeter(total=-5.0, count=2, max_value=2.0, min_value=-5.0, values=(-5.0, 2.0)),
        ]
    )
    assert meter.equal(
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )
    assert meter_merged.equal(
        ScalarMeter(
            total=129.0, count=17, max_value=8.0, min_value=-5.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_merge_():
    meter = ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    meter.merge_(
        [
            ScalarMeter(total=12.0, count=5, max_value=8.0, min_value=-1.0, values=(1.0, 3.0)),
            ScalarMeter(),
            ScalarMeter(total=-5.0, count=2, max_value=2.0, min_value=-5.0, values=(-5.0, 2.0)),
        ]
    )
    assert meter.equal(
        ScalarMeter(
            total=129.0, count=17, max_value=8.0, min_value=-5.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_meter_min():
    assert (
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).min()
        == -2.0
    )


def test_scalar_meter_min_empty():
    meter = ScalarMeter()
    with raises(EmptyMeterError):
        meter.min()


def test_scalar_meter_reset():
    meter = ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    meter.reset()
    assert meter.equal(ScalarMeter())


def test_scalar_meter_reset_empty():
    meter = ScalarMeter()
    meter.reset()
    assert meter.equal(ScalarMeter())


def test_scalar_meter_state_dict():
    assert ScalarMeter(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).state_dict() == {
        "count": 10,
        "total": 122.0,
        "values": (1.0, 3.0, 5.0, 4.0, 2.0),
        "max_value": 6.0,
        "min_value": -2.0,
    }


def test_scalar_meter_state_dict_empty():
    assert ScalarMeter().state_dict() == {
        "count": 0,
        "total": 0.0,
        "values": tuple(),
        "max_value": -float("inf"),
        "min_value": float("inf"),
    }


def test_scalar_meter_std():
    assert (
        ScalarMeter(
            total=10.0, count=10, max_value=1.0, min_value=1.0, values=(1.0, 1.0, 1.0, 1.0, 1.0)
        ).std()
        == 0.0
    )


def test_scalar_meter_std_empty():
    meter = ScalarMeter()
    with raises(EmptyMeterError):
        meter.std()


def test_scalar_meter_sum():
    assert (
        ScalarMeter(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).sum()
        == 122.0
    )


def test_scalar_meter_sum_empty():
    meter = ScalarMeter()
    with raises(EmptyMeterError):
        meter.sum()


def test_scalar_meter_update_4():
    meter = ScalarMeter()
    meter.update(4)
    assert meter.equal(ScalarMeter(count=1, total=4.0, min_value=4.0, max_value=4.0, values=(4.0,)))


def test_scalar_meter_update_4_and_2():
    meter = ScalarMeter()
    meter.update(4)
    meter.update(2)
    assert meter.equal(
        ScalarMeter(count=2, total=6.0, min_value=2.0, max_value=4.0, values=(4.0, 2.0))
    )


def test_scalar_meter_update_max_window_size_3():
    meter = ScalarMeter(max_size=3)
    meter.update(0)
    meter.update(3)
    meter.update(1)
    meter.update(4)
    assert meter.equal(
        ScalarMeter(count=4, total=8.0, min_value=0.0, max_value=4.0, values=(3.0, 1.0, 4.0))
    )


def test_scalar_meter_update_nan():
    meter = ScalarMeter()
    meter.update(float("NaN"))
    assert math.isnan(meter.total)
    assert meter.count == 1


def test_scalar_meter_update_inf():
    meter = ScalarMeter()
    meter.update(float("inf"))
    assert meter.equal(
        ScalarMeter(
            count=1,
            total=float("inf"),
            min_value=float("inf"),
            max_value=float("inf"),
            values=(float("inf"),),
        )
    )


@mark.parametrize("values", ([3, 1, 2], (3, 1, 2), (3.0, 1.0, 2.0)))
def test_scalar_meter_update_sequence(values: Union[list[float], tuple[float, ...]]):
    meter = ScalarMeter()
    meter.update_sequence(values)
    assert meter.equal(
        ScalarMeter(count=3, total=6.0, min_value=1.0, max_value=3.0, values=(3.0, 1.0, 2.0))
    )
