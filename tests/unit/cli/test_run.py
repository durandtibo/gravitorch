from typing import Any
from unittest.mock import patch

from objectory import OBJECT_TARGET
from omegaconf import OmegaConf

from gravitorch import constants as ct
from gravitorch.cli.run import main, run_cli
from gravitorch.runners import BaseRunner


class FakeRunner(BaseRunner):
    r"""Defines a fake runner to test the runner instantiation."""

    def run(self) -> Any:
        pass


def test_main() -> None:
    main({ct.RUNNER: {OBJECT_TARGET: "FakeRunner"}})


def test_main_factory_call() -> None:
    with patch("gravitorch.runners.BaseRunner.factory") as factory_mock:
        main({ct.RUNNER: {OBJECT_TARGET: "MyRunner", "engine": "ABC"}})
        factory_mock.assert_called_with(_target_="MyRunner", engine="ABC")


def test_run_cli_factory_call() -> None:
    with patch("gravitorch.runners.BaseRunner.factory") as factory_mock:
        run_cli(OmegaConf.create({ct.RUNNER: {OBJECT_TARGET: "MyRunner", "engine": "ABC"}}))
        factory_mock.assert_called_with(_target_="MyRunner", engine="ABC")
    assert OmegaConf.has_resolver("hya.add")
    assert OmegaConf.has_resolver("hya.mul")
