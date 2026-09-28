"""Read MODFLOW stress periods without altering their discretization.

Exchange rates are expressed in cubic metres per day, so a different declared
TDIS time unit must be rejected instead of being interpreted as days."""

from __future__ import annotations

import math
from pathlib import Path

from ..errors import ConfigurationError
from .text import _positive_int_text, _strip_comment
from .types import Mf6Period


def _parse_tdis(path: Path) -> tuple[str, tuple[Mf6Period, ...]]:
    time_units: str | None = None
    expected_nper: int | None = None
    periods: list[Mf6Period] = []
    inside_perioddata = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        upper = [token.upper() for token in tokens]
        if upper[0] == "TIME_UNITS" and len(tokens) >= 2:
            time_units = tokens[1].upper()
            continue
        if upper[0] == "NPER" and len(tokens) >= 2:
            expected_nper = _positive_int_text(tokens[1], f"NPER in {path}")
            continue
        if upper[:2] == ["BEGIN", "PERIODDATA"]:
            inside_perioddata = True
            continue
        if upper[:2] == ["END", "PERIODDATA"]:
            inside_perioddata = False
            continue
        if not inside_perioddata:
            continue
        if len(tokens) < 3:
            raise ConfigurationError(f"invalid PERIODDATA row in {path}: {raw_line}")
        try:
            perlen = float(tokens[0])
            nstp = int(tokens[1])
            tsmult = float(tokens[2])
        except ValueError as exc:
            raise ConfigurationError(
                f"invalid PERIODDATA row in {path}: {raw_line}"
            ) from exc
        if not math.isfinite(perlen) or perlen <= 0.0:
            raise ConfigurationError(f"PERLEN must be positive in {path}: {perlen}")
        if nstp < 1:
            raise ConfigurationError(f"NSTP must be at least one in {path}: {nstp}")
        if not math.isfinite(tsmult) or tsmult <= 0.0:
            raise ConfigurationError(f"TSMULT must be positive in {path}: {tsmult}")
        periods.append(Mf6Period(perlen_days=perlen, nstp=nstp, tsmult=tsmult))

    if time_units is None:
        raise ConfigurationError(f"TIME_UNITS was not found in {path}")
    if time_units != "DAYS":
        raise ConfigurationError(
            f"MF6 TIME_UNITS must be DAYS for coupling rates in m3/day, got {time_units} in {path}"
        )
    if expected_nper is None:
        raise ConfigurationError(f"NPER was not found in {path}")
    if len(periods) != expected_nper:
        raise ConfigurationError(
            f"TDIS declares NPER={expected_nper}, but {len(periods)} PERIODDATA rows were found in {path}"
        )
    return time_units, tuple(periods)
