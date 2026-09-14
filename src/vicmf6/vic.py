"""prepare, run, and accept restart-safe VIC Image Driver coupling windows."""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import VicConfig
from .errors import VicRuntimeError
from .exchange import ExchangeTable
from .mpi_spawn import spawn_vic
from .schedule import CouplingWindow, vic_record_count


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
    state_file: Path
    maximum_water_error_mm: float | None
    spawn_seconds: float
    output_read_seconds: float


class VicRuntime:
    """controller-owned VIC adapter with explicit restart and output ownership."""

    def __init__(
        self,
        config: VicConfig,
        exchange_table: ExchangeTable,
        *,
        logger: object,
    ) -> None:
        self.config = config
        self.exchange_table = exchange_table
        self.logger = logger
        self.template_text = config.global_file.read_text(encoding="utf-8")
        self.model_steps_per_day = _read_positive_int_parameter(
            self.template_text, "MODEL_STEPS_PER_DAY"
        )
        self.output_prefix = _resolve_exchange_output_prefix(
            self.template_text,
            exchange_variable=config.exchange_variable,
            configured_prefix=config.exchange_output_prefix,
        )
        self._latitude, self._longitude = self._resolve_vic_coordinates()

    def prepare_window(
        self,
        window: CouplingWindow,
        mapped_head_m: np.ndarray,
        *,
        previous_state: Path | None,
    ) -> PreparedVicWindow:
        """write the only mutable VIC inputs needed for this coupling interval."""

        records = vic_record_count(self.model_steps_per_day, window.duration_days)
        transformed_head = self.exchange_table.transform_head_for_vic(
            mapped_head_m, self.config.head_transform
        )

        window_tag = f"window-{window.index:04d}"
        output_directory = self.config.outputs_directory / window_tag
        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True, exist_ok=False)

        head_directory = self.config.exchange_directory / "heads"
        global_directory = self.config.exchange_directory / "globals"
        state_directory = self.config.exchange_directory / "states"
        for directory in (head_directory, global_directory, state_directory):
            directory.mkdir(parents=True, exist_ok=True)

        head_file = head_directory / f"{window_tag}.txt"
        _write_head_file(
            head_file,
            latitude=self._latitude,
            longitude=self._longitude,
            head_m=transformed_head,
        )

        state_prefix = state_directory / "state"
        expected_state = _state_path(state_prefix, window.end)
        if expected_state.exists():
            expected_state.unlink()

        global_file = global_directory / f"{window_tag}.global.txt"
        rendered = _render_global_parameter_file(
            self.template_text,
            window=window,
            nrecs=records,
            output_directory=output_directory,
            state_prefix=state_prefix,
            previous_state=previous_state,
        )
        global_file.write_text(rendered, encoding="utf-8")

        environment = dict(self.config.environment)
        environment.update(
            {
                "VIC_GW_FORMULATION": "head",
                "VIC_GW_HEAD_FILE": str(head_file),
                "VIC_GW_EXCHANGE_LENGTH_M": f"{self.config.exchange_length_m:.17g}",
                # the current validated vic source still names this internal
                # control a test scale. the public config uses a scientific
                # name so users do not need to know the development variable.
                "VIC_GW_TEST_KA_SCALE": f"{self.config.exchange_conductivity_scale:.17g}",
                # experiment d showed that a separate fractional numerical
                # limiter is unnecessary once only the physical liquid-water
                # availability bound remains active.
                "VIC_GW_MAX_DRAIN_FRACTION": "1.0",
                "VIC_GW_REFERENCE_DEPTH": "base",
            }
        )
        if not _is_whole_day(window.duration_days):
            environment["VIC_ALLOW_PARTIAL_DAY"] = "1"

        _log(
            self.logger,
            "info",
            f"prepared VIC {window_tag} records={records} head_min={np.min(transformed_head):.6f} head_max={np.max(transformed_head):.6f}",
        )
        return PreparedVicWindow(
            window=window,
            global_parameter_file=global_file,
            head_file=head_file,
            output_directory=output_directory,
            expected_state_file=expected_state,
            environment=environment,
        )

    def run_window(self, prepared: PreparedVicWindow) -> VicWindowResult:
        """spawn VIC, read the signed window amount, and accept only a fresh restart."""

        spawn_start = time.perf_counter()
        spawn_vic(
            executable=self.config.executable,
            working_directory=self.config.working_directory,
            global_parameter_file=prepared.global_parameter_file,
            mpi_processes=self.config.mpi_processes,
            omp_threads=self.config.omp_threads,
            timeout_seconds=self.config.spawn_timeout_seconds,
            preload_library=self.config.preload_library,
            environment=prepared.environment,
            logger=self.logger,
        )
        spawn_seconds = time.perf_counter() - spawn_start

        read_start = time.perf_counter()
        if not prepared.expected_state_file.is_file():
            matches = _state_candidates(
                prepared.expected_state_file.parent / "state", prepared.window.end
            )
            if len(matches) != 1:
                raise VicRuntimeError(
                    "VIC did not write the expected restart state: "
                    f"expected={prepared.expected_state_file} candidates={[str(path) for path in matches]}"
                )
            accepted_state = matches[0]
        else:
            accepted_state = prepared.expected_state_file

        exchange, water_error = _read_window_outputs(
            prepared.output_directory,
            prefix=self.output_prefix,
            exchange_variable=self.config.exchange_variable,
        )
        output_read_seconds = time.perf_counter() - read_start
        return VicWindowResult(
            exchange_grid_mm=exchange,
            state_file=accepted_state,
            maximum_water_error_mm=water_error,
            spawn_seconds=spawn_seconds,
            output_read_seconds=output_read_seconds,
        )

    def initial_state(self) -> Path | None:
        return self.config.initial_state

    def _resolve_vic_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        coordinates = self.exchange_table.vic_coordinates()
        if coordinates is not None:
            return coordinates

        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise VicRuntimeError(
                "exchange table lacks vic_lat/vic_lon and netCDF4 is unavailable to read VIC parameters"
            ) from exc

        with Dataset(str(self.config.parameters_file), "r") as dataset:
            if "lat" not in dataset.variables or "lon" not in dataset.variables:
                raise VicRuntimeError(
                    f"VIC parameters file does not contain lat/lon: {self.config.parameters_file}"
                )
            lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)

        rows = self.exchange_table.vic_rows()
        cols = self.exchange_table.vic_cols()
        if lat.ndim == 1 and lon.ndim == 1:
            if rows.max() >= lat.size or cols.max() >= lon.size:
                raise VicRuntimeError(
                    "exchange-table row/column exceeds VIC coordinate vectors"
                )
            return lat[rows], lon[cols]
        if lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
            if rows.max() >= lat.shape[0] or cols.max() >= lat.shape[1]:
                raise VicRuntimeError(
                    "exchange-table row/column exceeds VIC coordinate grids"
                )
            return lat[rows, cols], lon[rows, cols]
        raise VicRuntimeError(
            f"unsupported VIC coordinate shapes lat={lat.shape} lon={lon.shape}"
        )


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


def _read_window_outputs(
    output_directory: Path,
    *,
    prefix: str,
    exchange_variable: str,
) -> tuple[np.ndarray, float | None]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise VicRuntimeError(
            "netCDF4 is required to read VIC output; install vicmf6[runtime]"
        ) from exc

    paths = sorted(output_directory.glob(f"{prefix}.*.nc"))
    if not paths:
        raise VicRuntimeError(
            f"no VIC output files matching {prefix}.*.nc were written in {output_directory}"
        )

    exchange_total: np.ndarray | None = None
    maximum_water_error: float | None = None
    for path in paths:
        with Dataset(str(path), "r") as dataset:
            if exchange_variable not in dataset.variables:
                raise VicRuntimeError(f"{exchange_variable} was not found in {path}")
            exchange = _netcdf_array(dataset.variables[exchange_variable])
            exchange_2d = _sum_time_axis(exchange, exchange_variable, path)
            exchange_total = (
                exchange_2d.copy()
                if exchange_total is None
                else exchange_total + exchange_2d
            )

            if "OUT_WATER_ERROR" in dataset.variables:
                water_error = _netcdf_array(dataset.variables["OUT_WATER_ERROR"])
                finite = np.abs(water_error[np.isfinite(water_error)])
                if finite.size:
                    value = float(finite.max())
                    maximum_water_error = (
                        value
                        if maximum_water_error is None
                        else max(maximum_water_error, value)
                    )

    if exchange_total is None or exchange_total.ndim != 2:
        raise VicRuntimeError(
            "failed to construct a two-dimensional VIC exchange field"
        )
    return exchange_total, maximum_water_error


def _netcdf_array(variable: object) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        return np.asarray(values.filled(np.nan), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    for name in ("_FillValue", "missing_value"):
        if hasattr(variable, name):
            try:
                fill = float(getattr(variable, name))
                array = np.where(array == fill, np.nan, array)
            except (TypeError, ValueError):
                pass
    return array


def _sum_time_axis(array: np.ndarray, variable_name: str, path: Path) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return np.nansum(array, axis=0, dtype=np.float64)
    raise VicRuntimeError(
        f"{variable_name} has unsupported shape {array.shape} in {path}; expected (y,x) or (time,y,x)"
    )


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


def _seconds_since_midnight(timestamp: datetime) -> int:
    return timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second


def _is_whole_day(duration_days: float) -> bool:
    return np.isclose(duration_days, round(duration_days), atol=1.0e-12)


def _log(logger: object, level: str, message: str) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message)
