"""small immutable records shared by postprocessing modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class PostPaths:
    """all postprocessor-owned paths derived from one run directory."""

    root: Path
    tables: Path
    figures: Path
    report_markdown: Path
    summary_json: Path
    acceptance_json: Path
    acceptance_text: Path


@dataclass(frozen=True, slots=True)
class Mf6HeadSeries:
    """one MF6 model's head trajectory including the initial condition."""

    model_name: str
    times_days: np.ndarray
    heads_m: np.ndarray
    initial_heads_m: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.initial_heads_m.size)


@dataclass(frozen=True, slots=True)
class Mf6Geometry:
    """plotting/storage metadata for one MF6 model."""

    model_name: str
    node: np.ndarray
    area_m2: np.ndarray
    top_m: np.ndarray
    bottom_m: np.ndarray
    specific_storage_per_m: np.ndarray | None
    specific_yield: np.ndarray | None
    iconvert: np.ndarray | None
    x: np.ndarray | None
    y: np.ndarray | None
    row: np.ndarray | None
    col: np.ndarray | None
    vertices: tuple[tuple[tuple[float, float], ...], ...] | None
    source: str


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    """one named numerical acceptance result."""

    name: str
    passed: bool
    actual: float | int | str | None
    expected: float | int | str | None
    tolerance: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """aggregate acceptance state written by the postprocessor."""

    passed: bool
    checks: tuple[AcceptanceCheck, ...]
