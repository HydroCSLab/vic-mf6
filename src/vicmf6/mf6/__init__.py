"""Persistent MODFLOW adapter, sign conversion, and numerical diagnostics.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .boundary import api_rhs_from_rate as api_rhs_from_rate
from .lateral_flow import lateral_flow_diagnostics as lateral_flow_diagnostics
from .records import LateralFlowDiagnostics as LateralFlowDiagnostics
from .records import Mf6AdvanceResult as Mf6AdvanceResult
from .runtime import Mf6Runtime as Mf6Runtime
from .runtime import _log as _log
from .time_steps import _next_tdis_time_step_days as _next_tdis_time_step_days
from .time_steps import _strip_comment as _strip_comment

_sys.modules[__name__ + ".types"] = types
