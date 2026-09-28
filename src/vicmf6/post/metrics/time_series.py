"""Shared time alignment and signed aggregation rules for report tables."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ..records import Mf6HeadSeries


def cumulative_by_step(rows: list[dict[str, Any]], value_key: str) -> list[float]:
    total = 0.0
    values: list[float] = []
    for row in rows:
        total += float(row[value_key])
        values.append(total)
    return values


def _head_at_time(series: Mf6HeadSeries, time_days: float) -> np.ndarray | None:
    difference = np.abs(series.times_days - float(time_days))
    index = int(np.argmin(difference))
    if difference[index] > 1.0e-8:
        return None
    return np.asarray(series.heads_m[index], dtype=np.float64)


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


def _signed_from_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "positive_m3": float(np.sum(array[array > 0.0], dtype=np.float64)),
        "negative_m3": float(np.sum(array[array < 0.0], dtype=np.float64)),
        "net_m3": float(np.sum(array, dtype=np.float64)),
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


def _is_final_head_row(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    model = row["model"]
    maximum = max(float(item["time_days"]) for item in rows if item["model"] == model)
    return np.isclose(float(row["time_days"]), maximum, atol=1.0e-10)
