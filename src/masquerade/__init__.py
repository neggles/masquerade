try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from os import getenv

from rich.console import Console
from rich.traceback import install as _install_traceback

from .models import FrozenByT5Embedder, FrozenT5Embedder, VQAutoEncoder

is_debug = getenv("MASQUERADE_DEBUG", None) is not None

_ = _install_traceback(show_locals=is_debug, width=120, word_wrap=True)
console = Console(highlight=True)
