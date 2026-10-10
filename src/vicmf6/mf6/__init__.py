"""Persistent MODFLOW adapter, sign conversion, and numerical diagnostics."""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .boundary import api_rhs_from_rate as api_rhs_from_rate
from .lateral_flow import lateral_flow_diagnostics as lateral_flow_diagnostics
from .records import LateralFlowDiagnostics as LateralFlowDiagnostics
from .records import Mf6AdvanceResult as Mf6AdvanceResult
from .runtime import Mf6Runtime as Mf6Runtime

_sys.modules[__name__ + ".types"] = types
