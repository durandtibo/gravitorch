import torch
from pytest import mark

from gravitorch.nn import ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic, Sin
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


##############################
#     Tests for Gaussian     #
##############################


def test_gaussian_str() -> None:
    assert str(Gaussian()).startswith("Gaussian(")


@mark.parametrize("num_parameters", SIZES)
def test_gaussian_num_parameters(num_parameters: int):
    assert Gaussian(num_parameters).alpha.shape == (num_parameters,)


def test_gaussian_num_parameters_default() -> None:
    assert Gaussian().alpha.shape == (1,)


@mark.parametrize("init", (0.5, 1.0))
def test_gaussian_init(init: float):
    assert Gaussian(init=init).alpha.item() == init


def test_gaussian_learnable_true() -> None:
    assert Gaussian().alpha.requires_grad


def test_gaussian_learnable_false() -> None:
    assert not Gaussian(learnable=False).alpha.requires_grad


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_gaussian_forward(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    module = Gaussian(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_gaussian_forward_shape(device: str, shape: tuple[int, ...]):
    device = torch.device(device)
    module = Gaussian().to(device=device)
    assert module(torch.zeros(*shape, dtype=torch.float, device=device)).equal(
        torch.ones(*shape, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("num_parameters", (1, 4))
def test_gaussian_forward_num_parameters(device: str, num_parameters: int):
    device = torch.device(device)
    module = Gaussian(num_parameters).to(device=device)
    assert module(torch.zeros(2, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 4, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_gaussian_forward_init_1(device: str):
    device = torch.device(device)
    module = Gaussian().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.1353352832366127, 0.6065306597126334, 1.0, 0.6065306597126334, 0.1353352832366127],
            dtype=torch.float,
            device=device,
        )
    )


@mark.parametrize("device", get_available_devices())
def test_gaussian_forward_init_2(device: str):
    device = torch.device(device)
    module = Gaussian(init=2).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.6065306597126334, 0.8824969025845955, 1.0, 0.8824969025845955, 0.6065306597126334],
            dtype=torch.float,
            device=device,
        )
    )


############################
#     Tests for ExpSin     #
############################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_exp_sin_forward(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    module = ExpSin(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_exp_sin_forward_shape(device: str, shape: tuple[int, ...]):
    device = torch.device(device)
    module = ExpSin().to(device=device)
    assert module(torch.zeros(*shape, dtype=torch.float, device=device)).equal(
        torch.ones(*shape, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("num_parameters", (1, 4))
def test_exp_sin_forward_num_parameters(device: str, num_parameters: int):
    device = torch.device(device)
    module = ExpSin(num_parameters).to(device=device)
    assert module(torch.zeros(2, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 4, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_exp_sin_forward_init_1(device: str):
    device = torch.device(device)
    module = ExpSin().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.40280712612352804, 0.43107595064559234, 1.0, 2.319776824715853, 2.4825777280150008],
            dtype=torch.float,
            device=device,
        )
    )


@mark.parametrize("device", get_available_devices())
def test_exp_sin_forward_init_2(device: str):
    device = torch.device(device)
    module = ExpSin(init=2).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [2.1314499915144016, 0.40280712612352804, 1.0, 2.4825777280150008, 0.46916418587400077],
            dtype=torch.float,
            device=device,
        )
    )


#########################
#     Tests for Sin     #
#########################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_sin_forward(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    module = Sin(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_sin_forward_shape(device: str, shape: tuple[int, ...]):
    device = torch.device(device)
    module = Sin().to(device=device)
    assert module(torch.zeros(*shape, dtype=torch.float, device=device)).equal(
        torch.zeros(*shape, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("num_parameters", (1, 4))
def test_sin_forward_num_parameters(device: str, num_parameters: int):
    device = torch.device(device)
    module = Sin(num_parameters).to(device=device)
    assert module(torch.zeros(2, 4, dtype=torch.float, device=device)).equal(
        torch.zeros(2, 4, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_sin_forward_init_1(device: str):
    device = torch.device(device)
    module = Sin().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [-0.9092974268256817, -0.8414709848078965, 0.0, 0.8414709848078965, 0.9092974268256817],
            dtype=torch.float,
            device=device,
        )
    )


@mark.parametrize("device", get_available_devices())
def test_sin_forward_init_2(device: str):
    device = torch.device(device)
    module = Sin(init=2).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.7568024953079282, -0.9092974268256817, 0.0, 0.9092974268256817, -0.7568024953079282],
            dtype=torch.float,
            device=device,
        )
    )


###############################
#     Tests for Laplacian     #
###############################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_laplacian_forward(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    module = Laplacian(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_laplacian_forward_shape(device: str, shape: tuple[int, ...]):
    device = torch.device(device)
    module = Laplacian().to(device=device)
    assert module(torch.zeros(*shape, dtype=torch.float, device=device)).equal(
        torch.ones(*shape, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("num_parameters", (1, 4))
def test_laplacian_forward_num_parameters(device: str, num_parameters: int):
    device = torch.device(device)
    module = Laplacian(num_parameters).to(device=device)
    assert module(torch.zeros(2, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 4, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_laplacian_forward_init_1(device: str):
    device = torch.device(device)
    module = Laplacian().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.1353352832366127, 0.36787944117144233, 1.0, 0.36787944117144233, 0.1353352832366127],
            dtype=torch.float,
            device=device,
        )
    )


@mark.parametrize("device", get_available_devices())
def test_laplacian_forward_init_2(device: str):
    device = torch.device(device)
    module = Laplacian(init=2).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.36787944117144233, 0.6065306597126334, 1.0, 0.6065306597126334, 0.36787944117144233],
            dtype=torch.float,
            device=device,
        )
    )


###############################
#     Tests for Quadratic     #
###############################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_quadratic_forward(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    module = Quadratic(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_quadratic_forward_shape(device: str, shape: tuple[int, ...]):
    device = torch.device(device)
    module = Quadratic().to(device=device)
    assert module(torch.zeros(*shape, dtype=torch.float, device=device)).equal(
        torch.ones(*shape, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("num_parameters", (1, 4))
def test_quadratic_forward_num_parameters(device: str, num_parameters: int):
    device = torch.device(device)
    module = Quadratic(num_parameters).to(device=device)
    assert module(torch.zeros(2, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 4, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_quadratic_forward_init_1(device: str):
    device = torch.device(device)
    module = Quadratic().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(torch.tensor([0.2, 0.5, 1.0, 0.5, 0.2], dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
def test_quadratic_forward_init_2(device: str):
    device = torch.device(device)
    module = Quadratic(init=2).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.058823529411764705, 0.2, 1.0, 0.2, 0.058823529411764705],
            dtype=torch.float,
            device=device,
        )
    )


####################################
#     Tests for MultiQuadratic     #
####################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_multi_quadratic_forward(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    module = MultiQuadratic(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_multi_quadratic_forward_shape(device: str, shape: tuple[int, ...]):
    device = torch.device(device)
    module = MultiQuadratic().to(device=device)
    assert module(torch.zeros(*shape, dtype=torch.float, device=device)).equal(
        torch.ones(*shape, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("num_parameters", (1, 4))
def test_multi_quadratic_forward_num_parameters(device: str, num_parameters: int):
    device = torch.device(device)
    module = MultiQuadratic(num_parameters).to(device=device)
    assert module(torch.zeros(2, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 4, dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_multi_quadratic_forward_init_1(device: str):
    device = torch.device(device)
    module = MultiQuadratic().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.4472135954999579, 0.7071067811865475, 1.0, 0.7071067811865475, 0.4472135954999579],
            dtype=torch.float,
            device=device,
        )
    )


@mark.parametrize("device", get_available_devices())
def test_multi_quadratic_forward_init_2(device: str):
    device = torch.device(device)
    module = MultiQuadratic(init=2).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [0.24253562503633297, 0.4472135954999579, 1.0, 0.4472135954999579, 0.24253562503633297],
            dtype=torch.float,
            device=device,
        )
    )
