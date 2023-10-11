<p align="center">
    <a href="https://github.com/durandtibo/gravitorch/actions">
        <img alt="CI" src="https://github.com/durandtibo/gravitorch/workflows/CI/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/gravitorch/">
        <img alt="Documentation" src="https://github.com/durandtibo/gravitorch/workflows/Documentation/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/gravitorch/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/gravitorch/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/gravitorch/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/gravitorch/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/gravitorch">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/gravitorch/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/gravitorch/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/cbedbd2a20bf2a21cf22/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/gravitorch/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/cbedbd2a20bf2a21cf22/test_coverage" />
    </a>
    <br/>
    <a href="https://pypi.org/project/gravitorch/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/gravitorch">
    </a>
    <a href="https://pypi.org/project/gravitorch/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/gravitorch.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/gravitorch">
    </a>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <br/>
    <a href="https://pepy.tech/project/gravitorch">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/gravitorch">
    </a>
    <a href="https://pepy.tech/project/gravitorch">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/gravitorch/month">
    </a>
    <br/>
</p>

## Dependencies

The following is the corresponding `gravitorch` versions and supported dependencies.

| `gravitorch`              | `0.0.21`           |
|---------------------------|--------------------|
| `python`                  | `>=3.9,<3.12`      |
| `coola`                   | `>=0.0.20,<0.0.23` |
| `hya`                     | `>=0.0.12,<0.0.14` |
| `hydra-core`              | `>=1.3,<1.4`       |
| `minevent`                | `>=0.0.4,<0.0.5`   |
| `numpy`                   | `>=1.22,<1.26`     |
| `objectory`               | `>=0.0.7,<0.0.8`   |
| `pytorch-ignite`          | `>=0.4.11,<0.5`    |
| `tabulate`                | `>=0.9,<0.10`      |
| `torch`                   | `>=2.0,<2.1`       |
| `colorlog`<sup>*</sup>    | `>=6.7,<6.8`       |
| `matplotlib`<sup>*</sup>  | `>=3.6,<3.9`       |
| `pillow`<sup>*</sup>      | `>=9.0,<11.0`      |
| `psutil`<sup>*</sup>      | `>=5.9,<5.10`      |
| `startorch`<sup>*</sup>   | `>=0.0.5,<0.0.6`   |
| `tensorboard`<sup>*</sup> | `>=2.10,<2.15`     |
| `torchdata`<sup>*</sup>   | `>=0.6,<0.7`       |
| `tqdm`<sup>*</sup>        | `>=4.64,<4.67`     |

<sup>*</sup> indicates an optional dependency

## API stability

:warning: While `gravitorch` is in development stage, no API is guaranteed to be stable from one
release to the next. In fact, it is very likely that the API will change multiple times before a
stable 1.0.0 release. In practice, this means that upgrading `gravitorch` to a new version will
possibly break any code that was using the old version of `gravitorch`.

## License

`gravitorch` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](LICENSE) file.
