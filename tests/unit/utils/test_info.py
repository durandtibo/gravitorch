import logging
from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf
from pytest import LogCaptureFixture, fixture

from gravitorch.utils.info import log_run_info


@fixture
def config() -> DictConfig:
    return OmegaConf.create({"k": "v", "list": [1, {"a": "1", "b": "2"}]})


@patch("hydra.utils.get_original_cwd", lambda *args, **kwargs: "/my/path")
def test_log_run_info(caplog: LogCaptureFixture, config: DictConfig) -> None:
    caplog.set_level(logging.INFO)
    log_run_info(config)
    assert caplog.messages[-1].startswith("Config:")
