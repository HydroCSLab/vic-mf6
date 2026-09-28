"""Immutable metadata discovered from the two model input decks.

The native TDIS sequence is retained exactly. Coupling must meet its boundaries,
rather than silently changing groundwater timesteps to fit a requested window."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class VicOutputStream:
    """one VIC OUTFILE stream and the variables written to it."""

    prefix: str
    variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VicGlobalMetadata:
    """model metadata discovered from an immutable VIC global file."""

    global_file: Path
    working_directory: Path
    domain_file: Path
    parameters_file: Path
    forcing_prefixes: tuple[Path, ...]
    model_steps_per_day: int
    start_time: datetime
    end_time: datetime
    nrecs: int
    exchange_output_prefix: str
    initial_state: Path | None
    output_streams: tuple[VicOutputStream, ...]

    @property
    def duration_days(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 86400.0


@dataclass(frozen=True, slots=True)
class Mf6ModelMetadata:
    """one GWF model and the runtime identifiers discovered from its name file."""

    name: str
    namefile: Path
    api_package: str
    solution_id: int


@dataclass(frozen=True, slots=True)
class Mf6Period:
    """one MODFLOW 6 TDIS stress period."""

    perlen_days: float
    nstp: int
    tsmult: float


@dataclass(frozen=True, slots=True)
class Mf6SimulationMetadata:
    """simulation metadata discovered from mfsim.nam and TDIS."""

    namefile: Path
    workspace: Path
    tdis_file: Path
    time_units: str
    periods: tuple[Mf6Period, ...]
    models: tuple[Mf6ModelMetadata, ...]

    @property
    def total_time_days(self) -> float:
        return float(sum(period.perlen_days for period in self.periods))

    def time_step_boundaries_days(self) -> tuple[float, ...]:
        """return all solved MF6 time-step boundaries measured from time zero."""

        boundaries: list[float] = []
        elapsed = 0.0
        for period in self.periods:
            for dt in _period_time_steps(period):
                elapsed += dt
                boundaries.append(elapsed)
        return tuple(boundaries)


@dataclass(frozen=True, slots=True)
class _Mf6ModelEntry:
    name: str
    namefile: Path


def _period_time_steps(period: Mf6Period) -> tuple[float, ...]:
    if period.nstp == 1:
        return (period.perlen_days,)
    if math.isclose(period.tsmult, 1.0, rel_tol=0.0, abs_tol=1.0e-14):
        dt = period.perlen_days / period.nstp
        return tuple(dt for _ in range(period.nstp))

    denominator = period.tsmult**period.nstp - 1.0
    if math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ConfigurationError("MF6 TSMULT produced an invalid time-step denominator")
    first = period.perlen_days * (period.tsmult - 1.0) / denominator
    return tuple(first * period.tsmult**index for index in range(period.nstp))
