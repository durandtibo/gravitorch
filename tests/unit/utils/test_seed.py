import random
from unittest.mock import patch

import numpy as np
import torch
from coola import objects_are_equal
from pytest import raises

from gravitorch.utils.seed import (
    NumpyRandomSeedSetter,
    RandomRandomSeedSetter,
    RandomSeedSetter,
    TorchRandomSeedSetter,
    get_random_seed,
    get_torch_generator,
    manual_seed,
    numpy_seed,
    torch_seed,
)

#####################################
#     Tests for get_random_seed     #
#####################################


def test_get_random_seed() -> None:
    assert isinstance(get_random_seed(42), int)


def test_get_random_seed_same_seed() -> None:
    assert get_random_seed(42) == get_random_seed(42)


def test_get_random_seed_different_seeds() -> None:
    assert get_random_seed(1) != get_random_seed(42)


#########################################
#     Tests for get_torch_generator     #
#########################################


def test_get_torch_generator_same_seed() -> None:
    assert torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(1))
    )


def test_get_torch_generator_different_seeds() -> None:
    assert not torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(2))
    )


###########################################
#     Tests for NumpyRandomSeedSetter     #
###########################################


def test_numpy_random_seed_setter_str() -> None:
    assert str(NumpyRandomSeedSetter()).startswith("NumpyRandomSeedSetter(")


def test_numpy_random_seed_setter_manual_seed() -> None:
    seed_setter = NumpyRandomSeedSetter()
    seed_setter.manual_seed(42)
    x1 = np.random.randn(4, 6)
    x2 = np.random.randn(4, 6)
    seed_setter.manual_seed(42)
    x3 = np.random.randn(4, 6)
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


############################################
#     Tests for RandomRandomSeedSetter     #
############################################


def test_random_random_seed_setter_str() -> None:
    assert str(RandomRandomSeedSetter()).startswith("RandomRandomSeedSetter(")


def test_random_random_seed_setter_manual_seed() -> None:
    seed_setter = RandomRandomSeedSetter()
    seed_setter.manual_seed(42)
    x1 = random.uniform(0, 1)
    x2 = random.uniform(0, 1)
    seed_setter.manual_seed(42)
    x3 = random.uniform(0, 1)
    assert x1 == x3
    assert x1 != x2


###########################################
#     Tests for TorchRandomSeedSetter     #
###########################################


def test_torch_random_seed_setter_str() -> None:
    assert str(TorchRandomSeedSetter()).startswith("TorchRandomSeedSetter(")


def test_torch_random_seed_setter_manual_seed() -> None:
    seed_setter = TorchRandomSeedSetter()
    seed_setter.manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    seed_setter.manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
def test_torch_random_seed_setter_manual_seed_with_cuda() -> None:
    seed_setter = TorchRandomSeedSetter()
    with patch("torch.cuda.manual_seed_all") as mock_manual_seed_all:
        seed_setter.manual_seed(42)
        mock_manual_seed_all.assert_called_with(42)


######################################
#     Tests for RandomSeedSetter     #
######################################


def test_random_seed_setter_registered_setters() -> None:
    assert len(RandomSeedSetter.registry) == 3
    assert isinstance(RandomSeedSetter.registry["numpy"], NumpyRandomSeedSetter)
    assert isinstance(RandomSeedSetter.registry["random"], RandomRandomSeedSetter)
    assert isinstance(RandomSeedSetter.registry["torch"], TorchRandomSeedSetter)


def test_random_seed_setter_str() -> None:
    assert str(RandomSeedSetter()).startswith("RandomSeedSetter(")


def test_random_seed_setter_manual_seed() -> None:
    seed_setter = RandomSeedSetter()
    seed_setter.manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    seed_setter.manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


@patch.dict(RandomSeedSetter.registry, {}, clear=True)
def test_random_seed_setter_add_setter() -> None:
    assert len(RandomSeedSetter.registry) == 0
    RandomSeedSetter.add_setter("torch", TorchRandomSeedSetter())
    assert isinstance(RandomSeedSetter.registry["torch"], TorchRandomSeedSetter)


@patch.dict(RandomSeedSetter.registry, {}, clear=True)
def test_random_seed_setter_add_setter_exist_ok_false() -> None:
    assert len(RandomSeedSetter.registry) == 0
    RandomSeedSetter.add_setter("torch", TorchRandomSeedSetter())
    with raises(ValueError):
        RandomSeedSetter.add_setter("torch", TorchRandomSeedSetter())


@patch.dict(RandomSeedSetter.registry, {}, clear=True)
def test_random_seed_setter_add_setter_exist_ok_true() -> None:
    assert len(RandomSeedSetter.registry) == 0
    RandomSeedSetter.add_setter("torch", TorchRandomSeedSetter())
    RandomSeedSetter.add_setter("torch", NumpyRandomSeedSetter(), exist_ok=True)
    assert isinstance(RandomSeedSetter.registry["torch"], NumpyRandomSeedSetter)


def test_random_seed_setter_has_setter_true() -> None:
    assert RandomSeedSetter.has_setter("torch")


def test_random_seed_setter_has_setter_false() -> None:
    assert not RandomSeedSetter.has_setter("other")


#################################
#     Tests for manual_seed     #
#################################


def test_manual_seed_default() -> None:
    manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


def test_manual_seed_numpy_only() -> None:
    setter = NumpyRandomSeedSetter()
    manual_seed(42, setter)
    n1 = np.random.randn(4, 6)
    t1 = torch.randn(4, 6)
    manual_seed(42, setter)
    n2 = np.random.randn(4, 6)
    t2 = torch.randn(4, 6)
    assert np.array_equal(n1, n2)
    assert not t1.equal(t2)


################################
#     Tests for numpy_seed     #
################################


def test_numpy_seed_restore_random_seed() -> None:
    state = np.random.get_state()
    with numpy_seed(42):
        np.random.randn(4, 6)
    assert objects_are_equal(state, np.random.get_state())


def test_numpy_seed_restore_random_seed_with_exception() -> None:
    state = np.random.get_state()
    with raises(RuntimeError), numpy_seed(42):
        np.random.randn(4, 6)
        raise RuntimeError("Fake exception")
    assert objects_are_equal(state, np.random.get_state())


def test_numpy_seed_same_random_seed() -> None:
    with numpy_seed(42):
        x1 = np.random.randn(4, 6)
    with numpy_seed(42):
        x2 = np.random.randn(4, 6)
    assert np.array_equal(x1, x2)


################################
#     Tests for torch_seed     #
################################


def test_torch_seed_restore_random_seed() -> None:
    state = torch.get_rng_state()
    with torch_seed(42):
        torch.randn(4, 6)
    assert state.equal(torch.get_rng_state())


def test_torch_seed_restore_random_seed_with_exception() -> None:
    state = torch.get_rng_state()
    with raises(RuntimeError), torch_seed(42):
        torch.randn(4, 6)
        raise RuntimeError("Fake exception")
    assert state.equal(torch.get_rng_state())


def test_torch_seed_same_random_seed() -> None:
    with torch_seed(42):
        x1 = torch.randn(4, 6)
    with torch_seed(42):
        x2 = torch.randn(4, 6)
    assert x1.equal(x2)


def test_torch_seed_different_random_seeds() -> None:
    with torch_seed(42):
        x1 = torch.randn(4, 6)
    with torch_seed(142):
        x2 = torch.randn(4, 6)
    assert not x1.equal(x2)
