"""Offline exchange-table construction; callable and command-line APIs."""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .cli import main as main
from .intersections import build_exchange_table as build_exchange_table
from .mf6_geometry import load_mf6_cells as load_mf6_cells
from .records import BuildArtifacts as BuildArtifacts
from .records import ExchangeBuildError as ExchangeBuildError
from .records import Mf6SourceCell as Mf6SourceCell
from .records import VicSourceCell as VicSourceCell
from .summary import format_summary as format_summary
from .vic_geometry import load_vic_cells as load_vic_cells
from .vic_geometry import parse_vic_global as parse_vic_global

_sys.modules[__name__ + ".types"] = types
