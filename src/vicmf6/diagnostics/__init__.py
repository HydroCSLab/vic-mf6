"""Stable diagnostic schema, logging, and run-record writer."""

# The alias is package-qualified; plain import logging must resolve to Python.
import sys as _sys

from . import log_setup as logging
from .log_setup import build_logger as build_logger
from .records import WindowDiagnostics as WindowDiagnostics
from .records import make_window_diagnostics as make_window_diagnostics
from .writer import DiagnosticsWriter as DiagnosticsWriter

_sys.modules[__name__ + ".logging"] = logging
