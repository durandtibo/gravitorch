__all__ = ["__version__"]

from importlib.metadata import version

# from gravitorch import (
#     cli,
#     data,
#     distributed,
#     engines,
#     lr_schedulers,
#     models,
#     optimizers,
#     runners,
#     utils,
# )

__version__ = version(__name__)
