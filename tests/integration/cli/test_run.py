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
    assert tmp_path.joinpath(".hydra").is_dir()
    assert tmp_path.joinpath("run.log").is_file()


def test_run_cli_error(tmp_path: Path) -> None:
    with raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                r"python -m gravitorch.cli.run -cd=tests/integration/cli/conf "
                rf"-cn=simple num_classes=-1 hydra.run.dir={tmp_path}"
            ],
            shell=True,
            check=True,
        )
    assert tmp_path.joinpath(".hydra").is_dir()
    assert tmp_path.joinpath("run.log").is_file()
