"""Offline exchange-table construction; callable and command-line APIs.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .cli import _parser as _parser
from .cli import main as main
from .dependencies import _optional_imports as _optional_imports
from .intersections import _csv_value as _csv_value
from .intersections import build_exchange_table as build_exchange_table
from .mf6_geometry import _array_or_none as _array_or_none
from .mf6_geometry import _mf6_discretization as _mf6_discretization
from .mf6_geometry import _node_vertical_bounds as _node_vertical_bounds
from .mf6_geometry import _selected_surface_nodes as _selected_surface_nodes
from .mf6_geometry import load_mf6_cells as load_mf6_cells
from .records import BuildArtifacts as BuildArtifacts
from .records import ExchangeBuildError as ExchangeBuildError
from .records import Mf6SourceCell as Mf6SourceCell
from .records import VicSourceCell as VicSourceCell
from .summary import _coverage_stats as _coverage_stats
from .summary import _spacing_stats as _spacing_stats
from .summary import _stat_line as _stat_line
from .summary import _stats as _stats
from .summary import format_summary as format_summary
from .vic_geometry import _as_rectilinear_coordinates as _as_rectilinear_coordinates
from .vic_geometry import _coordinate_edges as _coordinate_edges
from .vic_geometry import _resolve_declared_path as _resolve_declared_path
from .vic_geometry import load_vic_cells as load_vic_cells
from .vic_geometry import parse_vic_global as parse_vic_global

_sys.modules[__name__ + ".types"] = types
