"""Render only the VIC directives owned by the coupling window.

The original template remains immutable. Removing owned directives before
appending replacements avoids ambiguous duplicate settings and keeps forcing
and physical model parameters under VIC ownership."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ..errors import VicRuntimeError
from ..schedule import CouplingWindow


def _render_global_parameter_file(
    template_text: str,
    *,
    window: CouplingWindow,
    nrecs: int,
    output_directory: Path,
    state_prefix: Path,
    previous_state: Path | None,
) -> str:
    owned = {
        "STARTYEAR",
        "STARTMONTH",
        "STARTDAY",
        "STARTSEC",
        "ENDYEAR",
        "ENDMONTH",
        "ENDDAY",
        "NRECS",
        "RESULT_DIR",
        "AGGFREQ",
        "INIT_STATE",
        "STATENAME",
        "STATEYEAR",
        "STATEMONTH",
        "STATEDAY",
        "STATESEC",
        "STATE_FORMAT",
    }
    preserved: list[str] = []
    for raw_line in template_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            preserved.append(raw_line)
            continue
        key = stripped.split()[0].upper()
        if key in owned:
            continue
        preserved.append(raw_line)

    state_seconds = _seconds_since_midnight(window.end)
    start_seconds = _seconds_since_midnight(window.start)
    aggregation = _aggregation_line(window.duration_days)

    overrides = [
        "",
        "#######################################################################",
        "# coupling runtime values",
        "#######################################################################",
        f"STARTYEAR   {window.start.year}",
        f"STARTMONTH  {window.start.month}",
        f"STARTDAY    {window.start.day}",
        f"STARTSEC    {start_seconds}",
        f"NRECS       {nrecs}",
        f"RESULT_DIR  {output_directory}",
        aggregation,
        f"STATENAME   {state_prefix}",
        f"STATEYEAR   {window.end.year}",
        f"STATEMONTH  {window.end.month}",
        f"STATEDAY    {window.end.day}",
        f"STATESEC    {state_seconds}",
        "STATE_FORMAT NETCDF4_CLASSIC",
    ]
    if previous_state is not None:
        overrides.append(f"INIT_STATE  {previous_state}")
    return "\n".join(preserved + overrides) + "\n"


def _resolve_exchange_output_prefix(
    template_text: str,
    *,
    exchange_variable: str,
    configured_prefix: str | None,
) -> str:
    streams: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in template_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()
        if key == "OUTFILE" and len(parts) >= 2:
            current = Path(parts[1]).name
            streams.setdefault(current, [])
        elif key == "OUTVAR" and len(parts) >= 2 and current is not None:
            streams[current].append(parts[1].upper())

    if configured_prefix is not None:
        prefix = Path(configured_prefix).name
        if prefix not in streams:
            raise VicRuntimeError(
                f"configured VIC exchange_output_prefix {prefix!r} was not found; streams={list(streams)}"
            )
        if exchange_variable.upper() not in streams[prefix]:
            raise VicRuntimeError(
                f"VIC stream {prefix!r} does not declare {exchange_variable}"
            )
        return prefix

    matches = [
        prefix
        for prefix, variables in streams.items()
        if exchange_variable.upper() in variables
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise VicRuntimeError(
            f"{exchange_variable} was not found under any OUTFILE in the VIC global template"
        )
    raise VicRuntimeError(
        f"{exchange_variable} appears in multiple VIC output streams {matches}; set vic.exchange_output_prefix"
    )


def _read_positive_int_parameter(template_text: str, key: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s+(\d+)\b", re.IGNORECASE)
    for raw_line in template_text.splitlines():
        if raw_line.lstrip().startswith("#"):
            continue
        match = pattern.match(raw_line.split("#", 1)[0])
        if match:
            value = int(match.group(1))
            if value >= 1:
                return value
    raise VicRuntimeError(f"{key} was not found in the VIC global parameter template")


def _aggregation_line(duration_days: float) -> str:
    rounded_days = int(round(duration_days))
    if np.isclose(duration_days, rounded_days, atol=1.0e-12) and rounded_days >= 1:
        return f"AGGFREQ     NDAYS {rounded_days}"
    hours = duration_days * 24.0
    rounded_hours = int(round(hours))
    if np.isclose(hours, rounded_hours, atol=1.0e-10) and rounded_hours >= 1:
        return f"AGGFREQ     NHOURS {rounded_hours}"
    raise VicRuntimeError(
        f"coupling window {duration_days} days cannot be represented as an integer number of VIC output hours"
    )


def _seconds_since_midnight(timestamp: datetime) -> int:
    return timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second


def _is_whole_day(duration_days: float) -> bool:
    return np.isclose(duration_days, round(duration_days), atol=1.0e-12)
