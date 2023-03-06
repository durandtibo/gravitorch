import os
import subprocess
from pathlib import Path

from pytest import raises


def test_run_cli_successful(tmp_path: Path) -> None:
    subprocess.run(
        [
            (
                "python -m gravitorch.cli.run -cd=tests/integration/cli/conf -cn=simple "
                f"hydra.run.dir={tmp_path}"
            )
        ],
        shell=True,
        check=True,
    )
    assert os.path.isdir(os.path.join(tmp_path, ".hydra"))
    assert os.path.exists(os.path.join(tmp_path, "run.log"))


def test_run_cli_error() -> None:
    with raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                r"python -m gravitorch.cli.run -cd=tests/integration/cli/conf -cn=simple num_classes=-1"
            ],
            shell=True,
            check=True,
        )
