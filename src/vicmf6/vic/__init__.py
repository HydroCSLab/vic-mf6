"""VIC adapter and stable window-result types."""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .records import PreparedVicWindow as PreparedVicWindow
from .records import VicWindowResult as VicWindowResult
from .runtime import VicRuntime as VicRuntime

_sys.modules[__name__ + ".types"] = types
