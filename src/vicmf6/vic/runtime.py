"""Controller-owned VIC window lifecycle.

Each window receives the groundwater head at its start, advances exactly once,
and returns an integrated exchange plus a fresh restart. File formatting and
NetCDF interpretation live in separate modules so this lifecycle stays visible."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import VicConfig
from ..errors import VicRuntimeError
from ..exchange import ExchangeTable
from ..mpi_spawn import spawn_vic
from ..schedule import CouplingWindow, vic_record_count
from .global_file import (
    _is_whole_day,
    _read_positive_int_parameter,
    _render_global_parameter_file,
    _resolve_exchange_output_prefix,
)
from .outputs import _read_window_outputs
from .restart_files import _state_candidates, _state_path, _write_head_file
from .types import PreparedVicWindow, VicWindowResult


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
            raise VicRuntimeError(f"VIC restart already exists: {expected_state}")

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


def _log(logger: object, level: str, message: str) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message)
