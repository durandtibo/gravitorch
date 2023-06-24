import logging

from coola import AllCloseTester, EqualityTester
from pytest import LogCaptureFixture

from gravitorch.utils.history.comparator import (
    BaseComparator,
    ComparatorAllCloseOperator,
    ComparatorEqualityOperator,
    MaxScalarComparator,
    MinScalarComparator,
)

#########################################
#     Tests for MaxScalarComparator     #
#########################################


def test_max_scalar_equal_true() -> None:
    assert MaxScalarComparator().equal(MaxScalarComparator())


def test_max_scalar_equal_false() -> None:
    assert not MaxScalarComparator().equal(MinScalarComparator())


def test_max_scalar_get_initial_best_value() -> None:
    assert MaxScalarComparator().get_initial_best_value() == -float("inf")


def test_max_scalar_is_better_int() -> None:
    comparator = MaxScalarComparator()
    assert comparator.is_better(5, 12)
    assert comparator.is_better(12, 12)
    assert not comparator.is_better(12, 5)


def test_max_scalar_is_better_float() -> None:
    comparator = MaxScalarComparator()
    assert comparator.is_better(5.2, 12.1)
    assert comparator.is_better(5.2, 5.2)
    assert not comparator.is_better(12.2, 5.1)


#########################################
#     Tests for MinScalarComparator     #
#########################################


def test_min_scalar_equal_true() -> None:
    assert MinScalarComparator().equal(MinScalarComparator())


def test_min_scalar_equal_false() -> None:
    assert not MinScalarComparator().equal(MaxScalarComparator())


def test_min_scalar_get_initial_best_value() -> None:
    assert MinScalarComparator().get_initial_best_value() == float("inf")


def test_min_scalar_is_better_int() -> None:
    comparator = MinScalarComparator()
    assert not comparator.is_better(5, 12)
    assert comparator.is_better(12, 12)
    assert comparator.is_better(12, 5)


def test_min_scalar_is_better_float() -> None:
    comparator = MinScalarComparator()
    assert not comparator.is_better(5.2, 12.1)
    assert comparator.is_better(5.2, 5.2)
    assert comparator.is_better(12.2, 5.1)


################################################
#     Tests for ComparatorAllCloseOperator     #
################################################


def test_comparator_allclose_operator_str() -> None:
    assert str(ComparatorAllCloseOperator()) == "ComparatorAllCloseOperator()"


def test_comparator_allclose_operator__eq__true() -> None:
    assert ComparatorAllCloseOperator() == ComparatorAllCloseOperator()


def test_comparator_allclose_operator__eq__false() -> None:
    assert ComparatorAllCloseOperator() != 123


def test_comparator_allclose_operator_equal_true() -> None:
    assert ComparatorAllCloseOperator().allclose(
        AllCloseTester(), MaxScalarComparator(), MaxScalarComparator()
    )


def test_comparator_allclose_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert ComparatorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=MaxScalarComparator(),
            object2=MaxScalarComparator(),
            show_difference=True,
        )
        assert not caplog.messages


def test_comparator_allclose_operator_equal_false_different_value() -> None:
    assert not ComparatorAllCloseOperator().allclose(
        AllCloseTester(), MaxScalarComparator(), MinScalarComparator()
    )


def test_comparator_allclose_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ComparatorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=MaxScalarComparator(),
            object2=MinScalarComparator(),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("`BaseComparator` objects are different")


def test_comparator_allclose_operator_equal_false_different_type() -> None:
    assert not ComparatorAllCloseOperator().allclose(AllCloseTester(), MaxScalarComparator(), 42)


def test_comparator_allclose_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ComparatorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=MaxScalarComparator(),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a `BaseComparator` object")


def test_comparator_allclose_operator_clone() -> None:
    op = ComparatorAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


################################################
#     Tests for ComparatorEqualityOperator     #
################################################


def test_comparator_equality_operator_str() -> None:
    assert str(ComparatorEqualityOperator()) == "ComparatorEqualityOperator()"


def test_comparator_equality_operator__eq__true() -> None:
    assert ComparatorEqualityOperator() == ComparatorEqualityOperator()


def test_comparator_equality_operator__eq__false() -> None:
    assert ComparatorEqualityOperator() != 123


def test_comparator_equality_operator_clone() -> None:
    op = ComparatorEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_comparator_equality_operator_equal_true() -> None:
    assert ComparatorEqualityOperator().equal(
        EqualityTester(), MaxScalarComparator(), MaxScalarComparator()
    )


def test_comparator_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert ComparatorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=MaxScalarComparator(),
            object2=MaxScalarComparator(),
            show_difference=True,
        )
        assert not caplog.messages


def test_comparator_equality_operator_equal_false_different_value() -> None:
    assert not ComparatorEqualityOperator().equal(
        EqualityTester(), MaxScalarComparator(), MinScalarComparator()
    )


def test_comparator_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ComparatorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=MaxScalarComparator(),
            object2=MinScalarComparator(),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("`BaseComparator` objects are different")


def test_comparator_equality_operator_equal_false_different_type() -> None:
    assert not ComparatorEqualityOperator().equal(EqualityTester(), MaxScalarComparator(), 42)


def test_comparator_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ComparatorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=MaxScalarComparator(),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a `BaseComparator` object")


def test_registered_operators() -> None:
    assert isinstance(AllCloseTester.registry[BaseComparator], ComparatorAllCloseOperator)
    assert isinstance(EqualityTester.registry[BaseComparator], ComparatorEqualityOperator)
