"""Stable diagnostic schema, logging, and run-record writer.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

# The alias is package-qualified; plain import logging must resolve to Python.
import sys as _sys

from . import log_setup as logging
from .log_setup import build_logger as build_logger
from .records import WindowDiagnostics as WindowDiagnostics
from .records import make_window_diagnostics as make_window_diagnostics
from .writer import DiagnosticsWriter as DiagnosticsWriter
from .writer import _package_version as _package_version
from .writer import _relative_net_error as _relative_net_error
from .writer import _signed_difference as _signed_difference
from .writer import _sum_signed as _sum_signed

_sys.modules[__name__ + ".logging"] = logging
