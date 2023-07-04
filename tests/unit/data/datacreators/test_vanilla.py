import logging

from pytest import LogCaptureFixture

from gravitorch.data.datacreators import DataCreator

#################################
#     Tests for DataCreator     #
#################################


def test_data_creator_str() -> None:
    assert str(DataCreator([1, 2, 3, 4])).startswith("DataCreator(")


def test_data_creator_create() -> None:
    assert DataCreator([1, 2, 3, 4]).create() == [1, 2, 3, 4]


def test_data_creator_create_deepcopy_true() -> None:
    creator = DataCreator([1, 2, 3, 4], deepcopy=True)
    data = creator.create()
    assert data == [1, 2, 3, 4]
    data.append(5)
    assert creator.create() == [1, 2, 3, 4]


def test_data_creator_create_deepcopy_false() -> None:
    creator = DataCreator([1, 2, 3, 4])
    data = creator.create()
    assert data == [1, 2, 3, 4]
    data.append(5)
    assert creator.create() == [1, 2, 3, 4, 5]


def test_data_creator_create_cannot_deepcopy(caplog: LogCaptureFixture) -> None:
    creator = DataCreator((i for i in range(5)), deepcopy=True)
    with caplog.at_level(level=logging.WARNING):
        data = creator.create()
        assert list(data) == [0, 1, 2, 3, 4]
        caplog.messages[0].startswith("The data can not be deepcopied")
