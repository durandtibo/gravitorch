__all__ = ["NoRepeatRunner"]

from datetime import datetime
from pathlib import Path
from typing import Union

from gravitorch.runners.base import BaseRunner
from gravitorch.runners.utils import setup_runner
from gravitorch.utils.format import str_indent
from gravitorch.utils.io import save_text
from gravitorch.utils.path import sanitize_path


class NoRepeatRunner(BaseRunner):
    r"""Implements a runner that does not repeat a successful run.

    This runner logs if a run was successful. If a previous run was
    successful, this runner does not execute the logic again.

    Args:
        runner (``BaseRunner`` or dict): Specifies the runner or its
            configuration.
        path (``Path`` or str): Specifies the path where to log a
            successful run.
    """

    def __init__(self, runner: Union[Union[BaseRunner, dict]], path: Union[Path, str]):
        self._path = sanitize_path(path)
        self._success_path = self._path.joinpath("_GRAVITORCH_SUCCESS_")
        self._runner = setup_runner(runner)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  runner={str_indent(str(self._runner))},\n"
            f"  path={self._path},\n"
            ")"
        )

    def run(self) -> None:
        r"""Executes the logic of the runner if it was not successful
        before."""
        if not self._success_path.is_file():
            self._runner.run()
            save_text(str(datetime.now()), self._success_path)
