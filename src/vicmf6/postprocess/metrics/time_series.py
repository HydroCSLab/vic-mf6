"""Shared time alignment and signed aggregation rules for report tables."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from itertools import islice
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ..records import Mf6HeadSeries


def _head_at_time(series: Mf6HeadSeries, time_days: float) -> np.ndarray | None:
    insertion = int(np.searchsorted(series.times_days, time_days))
    candidates = range(
        max(0, insertion - 1), min(insertion + 1, len(series.times_days))
    )
    index = min(candidates, key=lambda i: abs(series.times_days[i] - time_days))
    if abs(series.times_days[index] - time_days) > 1.0e-8:
        return None
    return series.read_heads(index)


def _duration_days(row: dict[str, Any]) -> float:
    start = datetime.fromisoformat(str(row["start"]))
    end = datetime.fromisoformat(str(row["end"]))
    duration = (end - start).total_seconds() / 86400.0
    if duration <= 0.0:
        raise PostprocessingError(
            f"non-positive coupling-window duration in step {row['step']}"
        )
    return duration


def _elapsed_days(config: ApplicationConfig, timestamp: str) -> float:
    value = datetime.fromisoformat(str(timestamp))
    return (value - config.coupling.start_time).total_seconds() / 86400.0


def _direction(value: float) -> str:
    if value > 0.0:
        return "vic_to_mf6"
    if value < 0.0:
        return "mf6_to_vic"
    return "zero"


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _signed_from_values(values: Iterable[float]) -> dict[str, float]:
    # Bound reduction workspace even when the input table has millions of rows.
    values = iter(values)
    totals = np.zeros(3, dtype=np.float64)
    while (array := np.fromiter(islice(values, 4096), dtype=np.float64)).size:
        totals += (
            np.sum(array[array > 0.0], dtype=np.float64),
            np.sum(array[array < 0.0], dtype=np.float64),
            np.sum(array, dtype=np.float64),
        )
    return {
        "positive_m3": float(totals[0]),
        "negative_m3": float(totals[1]),
        "net_m3": float(totals[2]),
    }


def _signed_from_rows(
    rows: list[dict[str, Any]],
    positive_key: str,
    negative_key: str,
    net_key: str,
) -> dict[str, float]:
    return {
        "positive_m3": float(sum(float(row[positive_key]) for row in rows)),
        "negative_m3": float(sum(float(row[negative_key]) for row in rows)),
        "net_m3": float(sum(float(row[net_key]) for row in rows)),
    }
