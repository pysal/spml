import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import base, decomposition, ensemble, linear_model, search

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("spatialml")
