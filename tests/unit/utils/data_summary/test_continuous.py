from typing import Union

import torch
from pytest import mark

from gravitorch.utils.data_summary.continuous import prepare_quantiles

#######################################
#     Tests for prepare_quantiles     #
#######################################


@mark.parametrize("quantiles", ([0.2, 0.8], (0.2, 0.8)))
def test_prepare_quantiles_list_or_tuple(quantiles: Union[list[float], tuple[float, ...]]) -> None:
    assert prepare_quantiles(quantiles).equal(torch.tensor([0.2, 0.8], dtype=torch.float))


def test_prepare_quantiles_torch_tensor() -> None:
    assert prepare_quantiles(torch.tensor([0.2, 0.8])).equal(
        torch.tensor([0.2, 0.8], dtype=torch.float)
    )


def test_prepare_quantiles_torch_tensor_sort_values() -> None:
    assert prepare_quantiles(torch.tensor([0.8, 0.2])).equal(
        torch.tensor([0.2, 0.8], dtype=torch.float)
    )
