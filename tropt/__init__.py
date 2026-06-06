# Expose tropt.__version__, read from install metadata (pyproject) with a source-tree fallback.
import importlib.metadata as _metadata

try:
    __version__ = _metadata.version("tropt")
except _metadata.PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

del _metadata
