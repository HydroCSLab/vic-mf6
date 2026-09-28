"""Geometry records and output artifacts of offline exchange-table construction.

Polygons belong only to preprocessing; the coupled runtime consumes a compact
table of numeric overlap areas and explicit model-cell identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExchangeBuildError(RuntimeError):
    """Raised when preprocessing cannot construct a trustworthy exchange table."""


@dataclass(frozen=True, slots=True)
class VicSourceCell:
    vic_id: str
    row: int
    col: int
    area_m2: float
    latitude: float
    longitude: float
    polygon_lonlat: Any
    interface_elevation_m: float | None


@dataclass(frozen=True, slots=True)
class Mf6SourceCell:
    model: str
    node: int  # one-based local node number
    area_m2: float
    polygon: Any
    top_m: float | None
    bottom_m: float | None


@dataclass(frozen=True, slots=True)
class BuildArtifacts:
    exchange_table: Path
    summary_json: Path
    summary_text: Path
    overlap_rows: int
    vic_cells: int
    mf6_cells: int
