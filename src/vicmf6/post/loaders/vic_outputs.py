"""Reconstruct VIC window totals from saved NetCDF output.

The same missing-data convention used at runtime must be retained here so
postprocessing cannot silently accept a window the coupler would reject."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ...netcdf import read_netcdf_array as _netcdf_array
from ...netcdf import sum_vic_time_records


def load_vic_exchange_fields(
    config: ApplicationConfig,
    *,
    expected_windows: int,
) -> list[dict[str, Any]]:
    """read the signed VIC exchange field accumulated in each coupling window."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise PostprocessingError(
            "VIC postprocessing requires netCDF4; install vicmf6[post]"
        ) from exc

    prefix = config.vic.exchange_output_prefix
    if not prefix:
        raise PostprocessingError(
            "the VIC exchange output stream could not be resolved"
        )

    windows: list[dict[str, Any]] = []
    for index in range(expected_windows):
        tag = f"window-{index:04d}"
        directory = config.vic.outputs_directory / tag
        paths = sorted(directory.glob(f"{prefix}.*.nc"))
        if not paths:
            raise PostprocessingError(
                f"no VIC output matching {prefix}.*.nc was found for {tag}: {directory}"
            )

        total: np.ndarray | None = None
        water_error_max: float | None = None
        for path in paths:
            with Dataset(str(path), "r") as dataset:
                variable_name = config.vic.exchange_variable
                if variable_name not in dataset.variables:
                    raise PostprocessingError(
                        f"{variable_name} was not found in VIC output {path}"
                    )
                values = _netcdf_array(dataset.variables[variable_name])
                field = _sum_time_axis(values, variable_name, path)
                total = field.copy() if total is None else total + field

                if "OUT_WATER_ERROR" in dataset.variables:
                    residual = _netcdf_array(dataset.variables["OUT_WATER_ERROR"])
                    finite = np.abs(residual[np.isfinite(residual)])
                    if finite.size:
                        maximum = float(np.max(finite))
                        water_error_max = (
                            maximum
                            if water_error_max is None
                            else max(water_error_max, maximum)
                        )

        if total is None or total.ndim != 2:
            raise PostprocessingError(
                f"failed to build a 2-D VIC exchange field for {tag}"
            )
        windows.append(
            {
                "index": index,
                "tag": tag,
                "field_mm": total,
                "water_error_max_mm": water_error_max,
                "files": tuple(paths),
            }
        )
    return windows


def _sum_time_axis(array: np.ndarray, variable_name: str, path: Path) -> np.ndarray:
    return sum_vic_time_records(
        array, variable_name, path, error_type=PostprocessingError
    )
