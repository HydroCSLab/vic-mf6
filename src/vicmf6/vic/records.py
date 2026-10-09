"""Inputs and accepted outputs of one restart-linked VIC window.

Exchange is an accumulated signed depth in millimetres, not a daily rate.
The returned state file is the initial state of the next window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schedule import CouplingWindow


@dataclass(frozen=True, slots=True)
class PreparedVicWindow:
    """all files and environment required for one child VIC run."""

    window: CouplingWindow
    global_parameter_file: Path
    head_file: Path
    output_directory: Path
    expected_state_file: Path
    environment: dict[str, str]


@dataclass(frozen=True, slots=True)
class VicWindowResult:
    """signed exchange amount and accepted restart produced by one window."""

    exchange_grid_mm: np.ndarray
    runoff_grid_mm: np.ndarray | None
    state_file: Path
    maximum_water_error_mm: float | None
    spawn_seconds: float
    output_read_seconds: float
