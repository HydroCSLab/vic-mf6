"""postprocessing for completed VIC-MODFLOW 6 coupled runs."""

from __future__ import annotations

from .runner import run_postprocessing

__all__ = ["run_postprocessing"]

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types

_sys.modules[__name__ + ".types"] = types
