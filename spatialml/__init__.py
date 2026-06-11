import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import base, ensemble, linear_model, search

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("spatialml")
