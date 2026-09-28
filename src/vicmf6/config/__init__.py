"""Public configuration records and YAML loading entry point.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

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
from .validation import _bool as _bool
from .validation import _config_path as _config_path
from .validation import _environment as _environment
from .validation import _expand as _expand
from .validation import _float as _float
from .validation import _nonnegative_float as _nonnegative_float
from .validation import _optional_path as _optional_path
from .validation import _positive_float as _positive_float
from .validation import _positive_int as _positive_int
from .validation import _require_mapping as _require_mapping
from .validation import _required as _required
from .validation import _text as _text
from .validation import (
    _validate_cross_section_contract as _validate_cross_section_contract,
)
from .validation import _validate_paths as _validate_paths

_sys.modules[__name__ + ".types"] = types
