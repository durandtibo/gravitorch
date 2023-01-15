r"""IO-related utility functions."""

__all__ = [
    "load_json",
    "load_pickle",
    "load_text",
    "load_yaml",
    "save_json",
    "save_pickle",
    "save_pytorch",
    "save_text",
    "save_yaml",
]

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import yaml
from torch import Tensor

logger = logging.getLogger(__name__)


def numpy_array_yaml_representer(dumper: yaml.Dumper, data: Union[np.ndarray, Tensor]):
    r"""This function computes the yaml representation of a numpy array or
    torch tensor."""
    if data.ndim == 0:
        scalar = data.item()
        if isinstance(scalar, float):
            return dumper.represent_float(scalar)
        return dumper.represent_int(scalar)
    return dumper.represent_list(data.tolist())


def numpy_datetime_yaml_representer(dumper: yaml.Dumper, data: np.datetime64):
    r"""This function computes the yaml representation of a numpy
    datetime64."""
    return dumper.represent_str(str(data))


def numpy_float_yaml_representer(dumper: yaml.Dumper, data: np.floating):
    r"""This function computes the yaml representation of a numpy
    float64/float32."""
    return dumper.represent_float(float(data))


def numpy_int_yaml_representer(dumper: yaml.Dumper, data: np.integer):
    r"""This function computes the yaml representation of a numpy
    int64/int32."""
    return dumper.represent_int(int(data))


# Set up the yaml dumper to deal with some data types like PyTorch tensors.
yaml.Dumper.add_representer(np.ndarray, numpy_array_yaml_representer)
yaml.Dumper.add_representer(np.datetime64, numpy_datetime_yaml_representer)
yaml.Dumper.add_representer(np.float64, numpy_float_yaml_representer)
yaml.Dumper.add_representer(np.float32, numpy_float_yaml_representer)
yaml.Dumper.add_representer(np.int64, numpy_int_yaml_representer)
yaml.Dumper.add_representer(np.int32, numpy_int_yaml_representer)
yaml.Dumper.add_representer(Tensor, numpy_array_yaml_representer)


###############
#    Text     #
###############


def load_text(path: Path) -> str:
    r"""Reads the data from a given text file.

    Args:
        path (``pathlib.Path``): Specifies the path where to the text
            file.

    Returns:
        The data from the text file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import load_text
        >>> data = load_text(Path('/path/to/data.txt'))
    """
    logger.debug(f"read {path}")
    with open(path) as file:
        data = file.read()
    return data


def save_text(to_save: str, path: Path) -> None:
    r"""Saves the given data in a text file.

    Args:
        to_save: Specifies the data to write in a text file.
        path (``pathlib.Path``): Specifies the path where to write the
            text file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import save_text
        >>> save_text("abc", Path('/path/to/data.txt'))
    """
    logger.debug(f"write data in a text file: {path}")
    path.parents[0].mkdir(exist_ok=True, parents=True)
    # Save to tmp, then commit by moving the file in case the job gets
    # interrupted while writing the file
    tmp_path = path.parents[0].joinpath(f"{path.name}.tmp")
    with open(tmp_path, mode="w") as file:
        file.write(to_save)
    tmp_path.rename(path)


################
#     JSON     #
################


class TensorJSONEncoder(json.JSONEncoder):
    r"""Defines a custom json encoder to work with numpy arrays and tensors."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (Tensor, np.ndarray)):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)  # use the string representation


def load_json(path: Path) -> dict:
    r"""Loads the data from a given json file.

    Args:
        path (``pathlib.Path``): Specifies the path to the json file.

    Returns:
        dict: The data from the json file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import load_json
        >>> data = load_json(Path('/path/to/data.json'))
    """
    logger.debug(f"Loading data from {path}...")
    with open(path, mode="rb") as file:
        data = json.load(file)
    return data


def save_json(to_save: Any, path: Path) -> None:
    r"""Saves the given data in a json file.

    Note: the keys are automatically converted to a string.

    Args:
        to_save: Specifies the data to write in a json file. The data
            should be JSON compatible.
        path (``pathlib.Path``): Specifies the path where to write the
            json file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import save_json
        >>> save_json({"key": "value"}, Path('/path/to/data.json'))
    """
    logger.debug(f"Saving data in a JSON file: {path}")
    path.parents[0].mkdir(exist_ok=True, parents=True)
    # Save to tmp, then commit by moving the file in case the job gets
    # interrupted while writing the file
    tmp_path = path.parents[0].joinpath(f"{path.name}.tmp")
    with open(tmp_path, "w") as file:
        json.dump(to_save, file, sort_keys=False, cls=TensorJSONEncoder)
    tmp_path.rename(path)


##################
#     Pickle     #
##################


def load_pickle(path: Path) -> Any:
    r"""Loads the data from a given pickle file.

    Args:
        path (``pathlib.Path``): Specifies the path to the pickle
            file.

    Returns:
        The data from the pickle file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import load_pickle
        >>> data = load_pickle(Path('/path/to/data.pkl'))
    """
    logger.debug(f"Loading data from {path}...")
    with open(path, mode="rb") as file:
        data = pickle.load(file)
    return data


def save_pickle(to_save: Any, path: Path, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    r"""Saves the given data in a pickle file.

    Args:
        to_save: Specifies the data to write in a pickle file.
        path (``pathlib.Path``): Specifies the path where to write
            the pickle file.
        protocol (int, optional): Specifies the pickle protocol. By
            default, it uses the highest protocol available.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import save_pickle
        >>> save_pickle({"key": "value"}, Path('/path/to/data.pkl'))
    """
    logger.debug(f"Saving data in pickle file: {path}  (protocol={protocol})")
    path.parents[0].mkdir(exist_ok=True, parents=True)
    # Save to tmp, then commit by moving the file in case the job gets
    # interrupted while writing the file
    tmp_path = path.parents[0].joinpath(f"{path.name}.tmp")
    with open(tmp_path, mode="wb") as file:
        pickle.dump(to_save, file, protocol=protocol)
    tmp_path.rename(path)


###################
#     PyTorch     #
###################


def save_pytorch(to_save: Any, path: Path) -> None:
    r"""Saves the data in a PyTorch file.

    Args:
        to_save: Specifies the data to write in the PyTorch file.
        path (``pathlib.Path``): Specifies the path where to write the
            PyTorch file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import save_pytorch
        >>> save_pytorch({"key": "value"}, Path('/path/to/data.pt'))
    """
    logger.debug(f"Saving data in a PyTorch file: {path}")
    path.parents[0].mkdir(exist_ok=True, parents=True)
    # Save to tmp, then commit by moving the file in case the job gets
    # interrupted while writing the file
    tmp_path = path.parents[0].joinpath(f"{path.name}.tmp")
    torch.save(to_save, tmp_path)
    tmp_path.rename(path)


################
#     YAML     #
################


def load_yaml(path: Path) -> Any:
    r"""Loads the data from a given yaml file.

    Args:
        path (``pathlib.Path``): Specifies the path to the yaml file.

    Returns:
        The data from the yaml file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import load_yaml
        >>> data = load_yaml(Path('/path/to/data.yaml'))
    """
    logger.debug(f"Loading data from {path}...")
    with open(path, mode="rb") as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(to_save: Any, path: Path) -> None:
    r"""Saves the data in a yaml file.

    Args:
        to_save: Specifies the data to write in a yaml file.
        path (``pathlib.Path``): Specifies the path where to write the
            yaml file.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from gravitorch.utils.io import save_yaml
        >>> save_yaml({"key": "value"}, Path('/path/to/data.yaml'))
    """
    logger.debug(f"Saving data in a yaml file: {path}")
    path.parents[0].mkdir(exist_ok=True, parents=True)
    # Save to tmp, then commit by moving the file in case the job gets
    # interrupted while writing the file
    tmp_path = path.parents[0].joinpath(f"{path.name}.tmp")
    with open(tmp_path, mode="w") as file:
        yaml.dump(to_save, file, Dumper=yaml.Dumper)
    tmp_path.rename(path)
