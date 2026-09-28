"""Read the VIC calendar, file references, and output stream declarations.

This parser does not run VIC or rewrite its template. The returned metadata
lets preflight check time compatibility before expensive native work begins."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path

from ..errors import ConfigurationError
from .text import (
    _nonnegative_int_text,
    _positive_int_text,
    _resolve_model_path,
    _strip_comment,
)
from .types import VicGlobalMetadata, VicOutputStream


def parse_vic_global(
    path: str | Path,
    *,
    exchange_variable: str = "OUT_GW_EXCHANGE",
) -> VicGlobalMetadata:
    """discover VIC model paths, calendar, native step, and exchange output stream."""

    global_file = Path(path).expanduser().resolve()
    if not global_file.is_file():
        raise ConfigurationError(f"VIC global file was not found: {global_file}")

    lines = global_file.read_text(encoding="utf-8").splitlines()
    entries = _vic_entries(lines)
    working_directory = global_file.parent

    domain_file = _resolve_model_path(
        working_directory,
        _vic_single(entries, "DOMAIN", global_file),
    )
    parameters_file = _resolve_model_path(
        working_directory,
        _vic_single(entries, "PARAMETERS", global_file),
    )

    forcing_prefixes: list[Path] = []
    for key in sorted(entries, key=_forcing_sort_key):
        if not re.fullmatch(r"FORCING\d+", key):
            continue
        value = _vic_single(entries, key, global_file)
        forcing_prefixes.append(_resolve_model_path(working_directory, value))
    if not forcing_prefixes:
        raise ConfigurationError(
            f"VIC global file does not declare a FORCING stream: {global_file}"
        )

    model_steps_per_day = _positive_int_text(
        _vic_single(entries, "MODEL_STEPS_PER_DAY", global_file),
        f"MODEL_STEPS_PER_DAY in {global_file}",
    )
    start_time = _vic_start_time(entries, global_file)
    nrecs, end_time = _vic_run_length(
        entries,
        global_file,
        start_time=start_time,
        model_steps_per_day=model_steps_per_day,
    )

    output_streams = _parse_vic_output_streams(lines)
    exchange_output_prefix = _find_exchange_output_prefix(
        output_streams,
        exchange_variable=exchange_variable,
        global_file=global_file,
    )

    initial_state = None
    if "INIT_STATE" in entries:
        value = _vic_single(entries, "INIT_STATE", global_file).strip()
        if value.upper() not in {"FALSE", "NONE", "NULL"}:
            initial_state = _resolve_model_path(working_directory, value)

    return VicGlobalMetadata(
        global_file=global_file,
        working_directory=working_directory,
        domain_file=domain_file,
        parameters_file=parameters_file,
        forcing_prefixes=tuple(forcing_prefixes),
        model_steps_per_day=model_steps_per_day,
        start_time=start_time,
        end_time=end_time,
        nrecs=nrecs,
        exchange_output_prefix=exchange_output_prefix,
        initial_state=initial_state,
        output_streams=output_streams,
    )


def _vic_entries(lines: list[str]) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split(maxsplit=1)
        if len(tokens) < 2:
            continue
        key = tokens[0].upper()
        entries.setdefault(key, []).append(tokens[1].strip())
    return entries


def _vic_single(
    entries: dict[str, list[str]],
    key: str,
    source: Path,
) -> str:
    values = entries.get(key, [])
    if not values:
        raise ConfigurationError(f"{key} was not found in VIC global file: {source}")
    if len(values) > 1:
        raise ConfigurationError(
            f"{key} is declared more than once in VIC global file: {source}"
        )
    return values[0]


def _vic_start_time(entries: dict[str, list[str]], source: Path) -> datetime:
    year = _positive_int_text(_vic_single(entries, "STARTYEAR", source), "STARTYEAR")
    month = _positive_int_text(_vic_single(entries, "STARTMONTH", source), "STARTMONTH")
    day = _positive_int_text(_vic_single(entries, "STARTDAY", source), "STARTDAY")
    start_seconds = 0
    if "STARTSEC" in entries:
        start_seconds = _nonnegative_int_text(
            _vic_single(entries, "STARTSEC", source), "STARTSEC"
        )
    if start_seconds >= 86400:
        raise ConfigurationError(
            f"STARTSEC must be less than 86400 in VIC global file: {source}"
        )
    try:
        start = datetime(year, month, day)
    except ValueError as exc:
        raise ConfigurationError(
            f"invalid VIC start calendar in global file {source}: {exc}"
        ) from exc
    return start + timedelta(seconds=start_seconds)


def _vic_run_length(
    entries: dict[str, list[str]],
    source: Path,
    *,
    start_time: datetime,
    model_steps_per_day: int,
) -> tuple[int, datetime]:
    """resolve VIC run length from either NRECS or the inclusive end date."""

    has_nrecs = "NRECS" in entries
    end_keys = ("ENDYEAR", "ENDMONTH", "ENDDAY")
    present_end_keys = [key for key in end_keys if key in entries]

    # vic accepts two equivalent ways to describe simulation length. the
    # source global file may provide a record count or an inclusive final
    # calendar day, but it must not provide both forms at once.
    if has_nrecs and present_end_keys:
        raise ConfigurationError(
            "VIC global file declares both NRECS and ENDYEAR/ENDMONTH/ENDDAY; "
            f"use only one run-length form: {source}"
        )

    if has_nrecs:
        nrecs = _positive_int_text(
            _vic_single(entries, "NRECS", source),
            f"NRECS in {source}",
        )
        seconds_per_step = 86400.0 / model_steps_per_day
        end_time = start_time + timedelta(seconds=nrecs * seconds_per_step)
        return nrecs, end_time

    if present_end_keys and len(present_end_keys) != len(end_keys):
        missing = [key for key in end_keys if key not in entries]
        raise ConfigurationError(
            "VIC global file has an incomplete end date; "
            f"missing {', '.join(missing)}: {source}"
        )

    if not present_end_keys:
        raise ConfigurationError(
            "VIC global file must declare either NRECS or "
            f"ENDYEAR/ENDMONTH/ENDDAY: {source}"
        )

    year = _positive_int_text(_vic_single(entries, "ENDYEAR", source), "ENDYEAR")
    month = _positive_int_text(_vic_single(entries, "ENDMONTH", source), "ENDMONTH")
    day = _positive_int_text(_vic_single(entries, "ENDDAY", source), "ENDDAY")
    try:
        final_day = datetime(year, month, day)
    except ValueError as exc:
        raise ConfigurationError(
            f"invalid VIC end calendar in global file {source}: {exc}"
        ) from exc

    # vic defines ENDYEAR/ENDMONTH/ENDDAY as the last calendar day that is
    # simulated. represent the resolved run internally as [start, end), so
    # the numerical end boundary is midnight immediately after that day.
    end_time = final_day + timedelta(days=1)
    if end_time <= start_time:
        raise ConfigurationError(
            f"VIC end date must be after the simulation start: {source}"
        )

    seconds_per_step = 86400.0 / model_steps_per_day
    record_count = (end_time - start_time).total_seconds() / seconds_per_step
    rounded = round(record_count)
    if not math.isclose(record_count, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ConfigurationError(
            "VIC start/end calendar does not contain an integer number of "
            f"model steps: {source}"
        )
    if rounded <= 0:
        raise ConfigurationError(f"VIC run contains no model records: {source}")
    return int(rounded), end_time


def _parse_vic_output_streams(lines: list[str]) -> tuple[VicOutputStream, ...]:
    streams: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        key = tokens[0].upper()
        if key == "OUTFILE" and len(tokens) >= 2:
            current = (Path(tokens[1]).name, [])
            streams.append(current)
            continue
        if key == "OUTVAR" and len(tokens) >= 2 and current is not None:
            current[1].append(tokens[1].upper())

    return tuple(
        VicOutputStream(prefix=prefix, variables=tuple(variables))
        for prefix, variables in streams
    )


def _find_exchange_output_prefix(
    streams: tuple[VicOutputStream, ...],
    *,
    exchange_variable: str,
    global_file: Path,
) -> str:
    target = exchange_variable.upper()
    matches = [stream.prefix for stream in streams if target in stream.variables]
    if not matches:
        raise ConfigurationError(
            f"{target} is not declared in any VIC OUTFILE stream: {global_file}"
        )
    if len(matches) > 1:
        raise ConfigurationError(
            f"{target} is declared in multiple VIC OUTFILE streams in {global_file}: {matches}"
        )
    return matches[0]


def _forcing_sort_key(key: str) -> tuple[int, str]:
    match = re.fullmatch(r"FORCING(\d+)", key)
    return (int(match.group(1)), key) if match else (10**9, key)
