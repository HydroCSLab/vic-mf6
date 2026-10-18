"""Public configuration records and YAML loading entry point."""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .directories import (
    create_fresh_run_output_directories as create_fresh_run_output_directories,
)
from .loading import load_config as load_config
from .provenance import config_as_dict as config_as_dict
from .records import ApplicationConfig as ApplicationConfig
from .records import CouplingConfig as CouplingConfig
from .records import DiagnosticsConfig as DiagnosticsConfig
from .records import Mf6Config as Mf6Config
from .records import VicConfig as VicConfig

_sys.modules[__name__ + ".types"] = types
