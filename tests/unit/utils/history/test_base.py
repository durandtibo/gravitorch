import logging

from coola import AllCloseTester, EqualityTester
from pytest import LogCaptureFixture

from gravitorch.utils.history import MaxScalarHistory, MinScalarHistory
from gravitorch.utils.history.base import (
    BaseHistory,
    HistoryAllCloseOperator,
    HistoryEqualityOperator,
)


def test_registered_operators() -> None:
    assert isinstance(AllCloseTester.registry[BaseHistory], HistoryAllCloseOperator)
    assert isinstance(EqualityTester.registry[BaseHistory], HistoryEqualityOperator)


#############################################
#     Tests for HistoryAllCloseOperator     #
#############################################


def test_history_allclose_operator_str() -> None:
    assert str(HistoryAllCloseOperator()) == "HistoryAllCloseOperator()"


def test_history_allclose_operator__eq__true() -> None:
    assert HistoryAllCloseOperator() == HistoryAllCloseOperator()


def test_history_allclose_operator__eq__false() -> None:
    assert HistoryAllCloseOperator() != 123


def test_history_allclose_operator_equal_true() -> None:
    assert HistoryAllCloseOperator().allclose(
        AllCloseTester(), MaxScalarHistory("accuracy"), MaxScalarHistory("accuracy")
    )


def test_history_allclose_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert HistoryAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=MaxScalarHistory("accuracy"),
            object2=MaxScalarHistory("accuracy"),
            show_difference=True,
        )
        assert not caplog.messages


def test_history_allclose_operator_equal_false_different_value() -> None:
    assert not HistoryAllCloseOperator().allclose(
        AllCloseTester(), MaxScalarHistory("accuracy"), MinScalarHistory("loss")
    )


def test_history_allclose_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not HistoryAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=MaxScalarHistory("accuracy"),
            object2=MinScalarHistory("loss"),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("`BaseHistory` objects are different")


def test_history_allclose_operator_equal_false_different_type() -> None:
    assert not HistoryAllCloseOperator().allclose(
        AllCloseTester(), MaxScalarHistory("accuracy"), 42
    )


def test_history_allclose_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not HistoryAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=MaxScalarHistory("accuracy"),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a `BaseHistory` object")


def test_history_allclose_operator_clone() -> None:
    op = HistoryAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


#############################################
#     Tests for HistoryEqualityOperator     #
#############################################


def test_history_equality_operator_str() -> None:
    assert str(HistoryEqualityOperator()) == "HistoryEqualityOperator()"


def test_history_equality_operator__eq__true() -> None:
    assert HistoryEqualityOperator() == HistoryEqualityOperator()


def test_history_equality_operator__eq__false() -> None:
    assert HistoryEqualityOperator() != 123


def test_history_equality_operator_clone() -> None:
    op = HistoryEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_history_equality_operator_equal_true() -> None:
    assert HistoryEqualityOperator().equal(
        EqualityTester(), MaxScalarHistory("accuracy"), MaxScalarHistory("accuracy")
    )


def test_history_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert HistoryEqualityOperator().equal(
            tester=EqualityTester(),
            object1=MaxScalarHistory("accuracy"),
            object2=MaxScalarHistory("accuracy"),
            show_difference=True,
        )
        assert not caplog.messages


def test_history_equality_operator_equal_false_different_value() -> None:
    assert not HistoryEqualityOperator().equal(
        EqualityTester(), MaxScalarHistory("accuracy"), MinScalarHistory("loss")
    )


def test_history_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not HistoryEqualityOperator().equal(
            tester=EqualityTester(),
            object1=MaxScalarHistory("accuracy"),
            object2=MinScalarHistory("loss"),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("`BaseHistory` objects are different")


def test_history_equality_operator_equal_false_different_type() -> None:
    assert not HistoryEqualityOperator().equal(EqualityTester(), MaxScalarHistory("accuracy"), 42)


def test_history_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not HistoryEqualityOperator().equal(
            tester=EqualityTester(),
            object1=MaxScalarHistory("accuracy"),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a `BaseHistory` object")
