from pytest import raises

from gravitorch.utils.meters import (
    EmptyMeterError,
    ExponentialMovingAverage,
    MovingAverage,
)

###################################
#     Tests for MovingAverage     #
###################################


def test_moving_average_repr() -> None:
    assert repr(MovingAverage()).startswith("MovingAverage(")


def test_moving_average_values() -> None:
    assert MovingAverage(values=(4, 2, 1)).values == (4, 2, 1)


def test_moving_average_values_empty() -> None:
    assert MovingAverage().values == ()


def test_moving_average_window_size() -> None:
    assert MovingAverage(window_size=5).window_size == 5


def test_moving_average_window_size_default() -> None:
    assert MovingAverage().window_size == 20


def test_moving_average_clone() -> None:
    meter = MovingAverage(values=(4, 2, 1), window_size=5)
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(MovingAverage(values=(4, 2, 1), window_size=5))


def test_moving_average_clone_empty() -> None:
    meter = MovingAverage()
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(MovingAverage())


def test_moving_average_equal_true() -> None:
    assert MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 1), window_size=5)
    )


def test_moving_average_equal_true_empty() -> None:
    assert MovingAverage().equal(MovingAverage())


def test_moving_average_equal_false_different_values() -> None:
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 2), window_size=5)
    )


def test_moving_average_equal_false_different_window_size() -> None:
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 1), window_size=10)
    )


def test_moving_average_equal_false_different_type() -> None:
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(1)


def test_moving_average_load_state_dict() -> None:
    meter = MovingAverage()
    meter.load_state_dict({"values": (4, 2, 1), "window_size": 5})
    assert meter.equal(MovingAverage(values=(4, 2, 1), window_size=5))


def test_moving_average_reset() -> None:
    meter = MovingAverage(values=(4, 2, 1), window_size=5)
    meter.reset()
    assert meter.equal(MovingAverage(window_size=5))


def test_moving_average_reset_empty() -> None:
    meter = MovingAverage()
    meter.reset()
    assert meter.equal(MovingAverage())


def test_moving_average_smoothed_average() -> None:
    assert MovingAverage(values=(4, 2)).smoothed_average() == 3.0


def test_moving_average_smoothed_average_empty() -> None:
    meter = MovingAverage()
    with raises(EmptyMeterError):
        meter.smoothed_average()


def test_moving_average_state_dict() -> None:
    assert MovingAverage(values=(4, 2, 1), window_size=5).state_dict() == {
        "values": (4, 2, 1),
        "window_size": 5,
    }


def test_moving_average_state_dict_empty() -> None:
    assert MovingAverage().state_dict() == {"values": (), "window_size": 20}


def test_moving_average_update() -> None:
    meter = MovingAverage()
    meter.update(4)
    meter.update(2)
    meter.equal(MovingAverage(values=(4, 2)))


##############################################
#     Tests for ExponentialMovingAverage     #
##############################################


def test_exponential_moving_average_repr() -> None:
    assert repr(ExponentialMovingAverage()).startswith("ExponentialMovingAverage(")


def test_exponential_moving_average_count() -> None:
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).count == 10


def test_exponential_moving_average_count_empty() -> None:
    assert ExponentialMovingAverage().count == 0


def test_exponential_moving_average_clone() -> None:
    meter = ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35))


def test_exponential_moving_average_clone_empty() -> None:
    meter = ExponentialMovingAverage()
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter_cloned.equal(ExponentialMovingAverage())


def test_exponential_moving_average_equal_true() -> None:
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_true_empty() -> None:
    assert ExponentialMovingAverage().equal(ExponentialMovingAverage())


def test_exponential_moving_average_equal_false_different_alpha() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.95, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_count() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=9, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_smoothed_average() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=2.35)
    )


def test_exponential_moving_average_equal_false_different_type() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(1)


def test_exponential_moving_average_load_state_dict() -> None:
    meter = ExponentialMovingAverage()
    meter.load_state_dict({"alpha": 0.9, "count": 10, "smoothed_average": 1.35})
    assert meter.equal(ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35))


def test_exponential_moving_average_reset() -> None:
    meter = ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    meter.reset()
    assert meter.equal(ExponentialMovingAverage(alpha=0.9))


def test_exponential_moving_average_reset_empty() -> None:
    meter = ExponentialMovingAverage()
    meter.reset()
    assert meter.equal(ExponentialMovingAverage())


def test_exponential_moving_average_state_dict() -> None:
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).state_dict() == {
        "alpha": 0.9,
        "count": 10,
        "smoothed_average": 1.35,
    }


def test_exponential_moving_average_state_dict_empty() -> None:
    assert ExponentialMovingAverage().state_dict() == {
        "alpha": 0.98,
        "count": 0,
        "smoothed_average": 0.0,
    }


def test_exponential_moving_average_smoothed_average() -> None:
    assert ExponentialMovingAverage(smoothed_average=1.35, count=1).smoothed_average() == 1.35


def test_exponential_moving_average_smoothed_average_empty() -> None:
    meter = ExponentialMovingAverage()
    with raises(EmptyMeterError):
        meter.smoothed_average()


def test_exponential_moving_average_update() -> None:
    meter = ExponentialMovingAverage()
    meter.update(4)
    meter.update(2)
    meter.equal(ExponentialMovingAverage(alpha=0.98, count=2, smoothed_average=3.96))
