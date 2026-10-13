import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import base, ensemble, linear_model, metrics, search, validation

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("spml")
