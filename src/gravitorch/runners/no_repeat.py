from __future__ import annotations

__all__ = ["NoRepeatRunner"]

from datetime import datetime
from pathlib import Path

from coola.utils import str_indent, str_mapping

from gravitorch.runners.base import BaseRunner
from gravitorch.runners.utils import setup_runner
from gravitorch.utils.io import save_text
from gravitorch.utils.path import sanitize_path


class NoRepeatRunner(BaseRunner):
    r"""Implements a runner that does not repeat a successful run.

    This runner logs if a run was successful. If a previous run was
    successful, this runner does not execute the logic again.

    Args:
    ----
        runner (``BaseRunner`` or dict): Specifies the runner or its
            configuration.
        path (``Path`` or str): Specifies the path where to log a
            successful run.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners import NoRepeatRunner, TrainingRunner
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> runner = NoRepeatRunner(TrainingRunner(engine), path="tmp/")
        >>> runner.run()
    """

    def __init__(self, runner: BaseRunner | dict, path: Path | str) -> None:
        self._path = sanitize_path(path)
        self._success_path = self._path.joinpath("_GRAVITORCH_SUCCESS_")
        self._runner = setup_runner(runner)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"runner": self._runner, "path": self._path}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def run(self) -> None:
        if not self._success_path.is_file():
            self._runner.run()
            save_text(str(datetime.now()), self._success_path)
