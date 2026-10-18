"""Write groundwater boundary files and locate exact VIC restart timestamps.

Coordinates identify cells independently of array order. Full precision in the
text boundary file avoids changing the physical head during serialization."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from ..errors import VicRuntimeError
from .global_file import _seconds_since_midnight


def _write_head_file(
    path: Path,
    *,
    latitude: Sequence[float],
    longitude: Sequence[float],
    head_m: Sequence[float],
) -> None:
    lat = np.asarray(latitude, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude, dtype=np.float64).reshape(-1)
    head = np.asarray(head_m, dtype=np.float64).reshape(-1)
    if not (lat.size == lon.size == head.size):
        raise VicRuntimeError(
            "VIC groundwater-head file vectors have different lengths"
        )
    if (
        not np.all(np.isfinite(lat))
        or not np.all(np.isfinite(lon))
        or not np.all(np.isfinite(head))
    ):
        raise VicRuntimeError("VIC groundwater-head file contains non-finite values")
    with path.open("w", encoding="utf-8") as stream:
        for lat_value, lon_value, head_value in zip(lat, lon, head, strict=True):
            stream.write(f"{lat_value:.17g} {lon_value:.17g} {head_value:.17g}\n")


def _state_path(prefix: Path, timestamp: datetime) -> Path:
    return Path(
        f"{prefix}.{timestamp.strftime('%Y%m%d')}_{_seconds_since_midnight(timestamp):05d}.nc"
    )


def _state_candidates(prefix: Path, timestamp: datetime) -> list[Path]:
    date = timestamp.strftime("%Y%m%d")
    seconds = _seconds_since_midnight(timestamp)
    candidates = [
        Path(f"{prefix}.{date}_{seconds:05d}.nc"),
        Path(f"{prefix}.{date}.nc")
        if seconds == 0
        else Path("/__vicmf6_no_candidate__"),
    ]
    return [path for path in candidates if path.is_file()]
