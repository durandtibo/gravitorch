from pytest import raises

from gravitorch.utils.meters import (
    EmptyMeterError,
    ExponentialMovingAverage,
    MovingAverage,
)

###################################
#     Tests for MovingAverage     #
###################################


def test_moving_average_repr():
    assert repr(MovingAverage()).startswith("MovingAverage(")


def test_moving_average_values():
    assert MovingAverage(values=(4, 2, 1)).values == (4, 2, 1)


def test_moving_average_values_empty():
    assert MovingAverage().values == tuple()


def test_moving_average_window_size():
    assert MovingAverage(window_size=5).window_size == 5


def test_moving_average_window_size_default():
    assert MovingAverage().window_size == 20


def test_moving_average_clone():
    meter = MovingAverage(values=(4, 2, 1), window_size=5)
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(MovingAverage(values=(4, 2, 1), window_size=5))


def test_moving_average_clone_empty():
    meter = MovingAverage()
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(MovingAverage())


def test_moving_average_equal_true():
    assert MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 1), window_size=5)
    )


def test_moving_average_equal_true_empty():
    assert MovingAverage().equal(MovingAverage())


def test_moving_average_equal_false_different_values():
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 2), window_size=5)
    )


def test_moving_average_equal_false_different_window_size():
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 1), window_size=10)
    )


def test_moving_average_equal_false_different_type():
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(1)


def test_moving_average_load_state_dict():
    meter = MovingAverage()
    meter.load_state_dict({"values": (4, 2, 1), "window_size": 5})
    assert meter.equal(MovingAverage(values=(4, 2, 1), window_size=5))


def test_moving_average_reset():
    meter = MovingAverage(values=(4, 2, 1), window_size=5)
    meter.reset()
    assert meter.equal(MovingAverage(window_size=5))


def test_moving_average_reset_empty():
    meter = MovingAverage()
    meter.reset()
    assert meter.equal(MovingAverage())


def test_moving_average_smoothed_average():
    assert MovingAverage(values=(4, 2)).smoothed_average() == 3.0


def test_moving_average_smoothed_average_empty():
    meter = MovingAverage()
    with raises(EmptyMeterError):
        meter.smoothed_average()


def test_moving_average_state_dict():
    assert MovingAverage(values=(4, 2, 1), window_size=5).state_dict() == {
        "values": (4, 2, 1),
        "window_size": 5,
    }


def test_moving_average_state_dict_empty():
    assert MovingAverage().state_dict() == {"values": tuple(), "window_size": 20}


def test_moving_average_update():
    meter = MovingAverage()
    meter.update(4)
    meter.update(2)
    meter.equal(MovingAverage(values=(4, 2)))


##############################################
#     Tests for ExponentialMovingAverage     #
##############################################


def test_exponential_moving_average_repr():
    assert repr(ExponentialMovingAverage()).startswith("ExponentialMovingAverage(")


def test_exponential_moving_average_count():
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).count == 10


def test_exponential_moving_average_count_empty():
    assert ExponentialMovingAverage().count == 0


def test_exponential_moving_average_clone():
    meter = ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35))


def test_exponential_moving_average_clone_empty():
    meter = ExponentialMovingAverage()
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(ExponentialMovingAverage())


def test_exponential_moving_average_equal_true():
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_true_empty():
    assert ExponentialMovingAverage().equal(ExponentialMovingAverage())


def test_exponential_moving_average_equal_false_different_alpha():
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.95, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_count():
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=9, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_smoothed_average():
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=2.35)
    )


def test_exponential_moving_average_equal_false_different_type():
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(1)


def test_exponential_moving_average_load_state_dict():
    meter = ExponentialMovingAverage()
    meter.load_state_dict({"alpha": 0.9, "count": 10, "smoothed_average": 1.35})
    assert meter.equal(ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35))


def test_exponential_moving_average_reset():
    meter = ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    meter.reset()
    assert meter.equal(ExponentialMovingAverage(alpha=0.9))


def test_exponential_moving_average_reset_empty():
    meter = ExponentialMovingAverage()
    meter.reset()
    assert meter.equal(ExponentialMovingAverage())


def test_exponential_moving_average_state_dict():
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).state_dict() == {
        "alpha": 0.9,
        "count": 10,
        "smoothed_average": 1.35,
    }


def test_exponential_moving_average_state_dict_empty():
    assert ExponentialMovingAverage().state_dict() == {
        "alpha": 0.98,
        "count": 0,
        "smoothed_average": 0.0,
    }


def test_exponential_moving_average_smoothed_average():
    assert ExponentialMovingAverage(smoothed_average=1.35, count=1).smoothed_average() == 1.35


def test_exponential_moving_average_smoothed_average_empty():
    meter = ExponentialMovingAverage()
    with raises(EmptyMeterError):
        meter.smoothed_average()


def test_exponential_moving_average_update():
    meter = ExponentialMovingAverage()
    meter.update(4)
    meter.update(2)
    meter.equal(ExponentialMovingAverage(alpha=0.98, count=2, smoothed_average=3.96))
