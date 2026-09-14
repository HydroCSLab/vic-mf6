"""read model-owned metadata from VIC and MODFLOW 6 input files."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .errors import ConfigurationError


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


def parse_mf6_simulation(path: str | Path) -> Mf6SimulationMetadata:
    """discover GWF models, API packages, solution groups, and TDIS metadata."""

    namefile = Path(path).expanduser().resolve()
    if not namefile.is_file():
        raise ConfigurationError(f"MF6 simulation name file was not found: {namefile}")

    workspace = namefile.parent
    lines = namefile.read_text(encoding="utf-8").splitlines()
    model_entries = _parse_mf6_model_entries(lines, workspace, namefile)
    solution_ids = _parse_mf6_solution_groups(lines, model_entries, namefile)
    tdis_file = _parse_tdis_path(lines, workspace, namefile)
    time_units, periods = _parse_tdis(tdis_file)

    models: list[Mf6ModelMetadata] = []
    for entry in model_entries:
        api_package = _parse_single_api_package(entry.namefile, entry.name)
        models.append(
            Mf6ModelMetadata(
                name=entry.name,
                namefile=entry.namefile,
                api_package=api_package,
                solution_id=solution_ids[entry.name],
            )
        )

    return Mf6SimulationMetadata(
        namefile=namefile,
        workspace=workspace,
        tdis_file=tdis_file,
        time_units=time_units,
        periods=periods,
        models=tuple(models),
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


def _parse_mf6_model_entries(
    lines: list[str],
    workspace: Path,
    source: Path,
) -> tuple[_Mf6ModelEntry, ...]:
    models: list[_Mf6ModelEntry] = []
    inside_models = False
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        upper = [token.upper() for token in tokens]
        if upper[:2] == ["BEGIN", "MODELS"]:
            inside_models = True
            continue
        if upper[:2] == ["END", "MODELS"]:
            inside_models = False
            continue
        if not inside_models or not upper[0].startswith("GWF"):
            continue
        if len(tokens) < 2:
            raise ConfigurationError(f"invalid GWF model entry in {source}: {raw_line}")
        model_namefile = _resolve_model_path(workspace, tokens[1])
        model_name = (
            tokens[2].upper() if len(tokens) >= 3 else model_namefile.stem.upper()
        )
        models.append(_Mf6ModelEntry(name=model_name, namefile=model_namefile))

    if not models:
        raise ConfigurationError(f"no GWF models were declared in {source}")
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ConfigurationError(f"duplicate GWF model names in {source}: {names}")
    for model in models:
        if not model.namefile.is_file():
            raise ConfigurationError(
                f"GWF name file was not found for model {model.name}: {model.namefile}"
            )
    return tuple(models)


def _parse_mf6_solution_groups(
    lines: list[str],
    models: tuple[_Mf6ModelEntry, ...],
    source: Path,
) -> dict[str, int]:
    known = {model.name for model in models}
    assignments: dict[str, list[int]] = {name: [] for name in known}
    current_group: int | None = None

    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        upper = [token.upper() for token in tokens]
        if upper[:2] == ["BEGIN", "SOLUTIONGROUP"]:
            if len(tokens) < 3:
                raise ConfigurationError(
                    f"SOLUTIONGROUP is missing its identifier in {source}: {raw_line}"
                )
            try:
                current_group = int(tokens[2])
            except ValueError as exc:
                raise ConfigurationError(
                    f"invalid SOLUTIONGROUP identifier in {source}: {tokens[2]}"
                ) from exc
            continue
        if upper[:2] == ["END", "SOLUTIONGROUP"]:
            current_group = None
            continue
        if current_group is None or not upper[0].startswith("IMS"):
            continue
        for model_name in upper[2:]:
            if model_name in assignments:
                assignments[model_name].append(current_group)

    resolved: dict[str, int] = {}
    for model_name in sorted(known):
        groups = assignments[model_name]
        if not groups:
            raise ConfigurationError(
                f"GWF model {model_name} is not assigned to a SOLUTIONGROUP in {source}"
            )
        unique = sorted(set(groups))
        if len(unique) != 1:
            raise ConfigurationError(
                f"GWF model {model_name} is assigned to multiple solution groups in {source}: {unique}"
            )
        resolved[model_name] = unique[0]
    return resolved


def _parse_tdis_path(lines: list[str], workspace: Path, source: Path) -> Path:
    candidates: list[Path] = []
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        if tokens[0].upper().startswith("TDIS") and len(tokens) >= 2:
            candidates.append(_resolve_model_path(workspace, tokens[1]))
    if len(candidates) != 1:
        raise ConfigurationError(
            f"expected exactly one TDIS entry in {source}, found {len(candidates)}"
        )
    tdis_file = candidates[0]
    if not tdis_file.is_file():
        raise ConfigurationError(f"TDIS file was not found: {tdis_file}")
    return tdis_file


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


def _parse_single_api_package(model_namefile: Path, model_name: str) -> str:
    packages: list[str] = []
    for raw_line in model_namefile.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        if not tokens[0].upper().startswith("API"):
            continue
        if len(tokens) < 3 or not tokens[2].strip():
            raise ConfigurationError(
                f"API6 package in {model_namefile} must have an explicit package name for XMI addressing"
            )
        packages.append(tokens[2].upper())

    if not packages:
        raise ConfigurationError(
            f"GWF model {model_name} does not declare an API6 package in {model_namefile}"
        )
    if len(packages) > 1:
        raise ConfigurationError(
            f"GWF model {model_name} declares multiple API6 packages in {model_namefile}: {packages}"
        )
    return packages[0]


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


def _resolve_model_path(base: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


def _forcing_sort_key(key: str) -> tuple[int, str]:
    match = re.fullmatch(r"FORCING(\d+)", key)
    return (int(match.group(1)), key) if match else (10**9, key)


def _positive_int_text(raw: str, field_name: str) -> int:
    try:
        value = int(raw.split()[0])
    except (ValueError, IndexError) as exc:
        raise ConfigurationError(f"{field_name} must be an integer") from exc
    if value < 1:
        raise ConfigurationError(f"{field_name} must be at least 1")
    return value


def _nonnegative_int_text(raw: str, field_name: str) -> int:
    try:
        value = int(raw.split()[0])
    except (ValueError, IndexError) as exc:
        raise ConfigurationError(f"{field_name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{field_name} must be nonnegative")
    return value


def _strip_comment(raw_line: str) -> str:
    return raw_line.split("#", 1)[0].strip()
