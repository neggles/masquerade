try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from rich.console import Console
from rich.traceback import install as _install_traceback

_orig_handler = _install_traceback(show_locals=True, width=120, word_wrap=True)
console = Console(highlight=True)
