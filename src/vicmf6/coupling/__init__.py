"""Explicit coupling: orchestration, physical stages, MPI, and reporting.

Start with explicit.run_coupling to read the algorithm. Native model operations
belong to vic/ and mf6/; grid transfer formulas belong to exchange/.
"""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types

_sys.modules[__name__ + ".types"] = types
