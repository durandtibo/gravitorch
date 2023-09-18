import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture
from torch.utils.data import Dataset

from gravitorch.datasets import log_box_dataset_class

###########################################
#     Tests for log_box_dataset_class     #
###########################################


def test_log_box_dataset_class(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        log_box_dataset_class(Mock(spec=Dataset))
        assert caplog.messages[0] == " ---------"
        assert caplog.messages[1] == "| Dataset |"
        assert caplog.messages[2] == " ---------"
