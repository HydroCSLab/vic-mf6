"""Read the accumulated exchange and water-error diagnostics from VIC NetCDF.

Missing values must survive summation. The exchange map later rejects missing
data on coupled cells while allowing the inactive raster outside the domain."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import VicRuntimeError
from ..netcdf import read_netcdf_array as _netcdf_array
from ..netcdf import sum_vic_time_records


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


def _sum_time_axis(array: np.ndarray, variable_name: str, path: Path) -> np.ndarray:
    return sum_vic_time_records(array, variable_name, path, error_type=VicRuntimeError)
