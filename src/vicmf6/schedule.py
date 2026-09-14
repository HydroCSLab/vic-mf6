"""construct explicit coupling windows without changing the requested scheme."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class CouplingWindow:
    """one accepted explicit exchange interval."""

    index: int
    start: datetime
    end: datetime

    @property
    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400.0


def build_windows(
    start_time: datetime,
    end_time: datetime,
    interval_days: float,
) -> list[CouplingWindow]:
    """split the half-open simulation interval [start_time, end_time) into windows."""

    if end_time <= start_time:
        raise ConfigurationError("coupling end time must be later than start time")
    if interval_days <= 0.0:
        raise ConfigurationError("coupling interval must be greater than zero")

    interval = timedelta(days=float(interval_days))
    windows: list[CouplingWindow] = []
    current = start_time
    index = 0
    while current < end_time:
        boundary = min(current + interval, end_time)
        windows.append(CouplingWindow(index=index, start=current, end=boundary))
        current = boundary
        index += 1
    return windows


def vic_record_count(model_steps_per_day: int, duration_days: float) -> int:
    """convert a coupling duration to an exact number of native VIC records."""

    if model_steps_per_day < 1:
        raise ConfigurationError("MODEL_STEPS_PER_DAY must be at least one")
    raw_count = model_steps_per_day * float(duration_days)
    rounded_count = int(round(raw_count))
    if rounded_count < 1 or not math.isclose(raw_count, rounded_count, abs_tol=1.0e-10):
        raise ConfigurationError(
            "coupling interval does not contain an integer number of VIC model steps: "
            f"steps_per_day={model_steps_per_day} duration_days={duration_days} records={raw_count}"
        )
    return rounded_count
