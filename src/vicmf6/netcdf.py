"""Shared NetCDF missing-data and temporal aggregation rules.

Runtime and postprocessing use exactly the same numerical interpretation of
VIC records. NetCDF itself is loaded only by the file-reading callers.
"""

from pathlib import Path

import numpy as np


def read_netcdf_array(variable: object) -> np.ndarray:
    """Convert NetCDF masks and declared fill values to explicit NaNs."""
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


def sum_vic_time_records(
    array: np.ndarray, variable_name: str, path: Path, *, error_type: type[Exception]
) -> np.ndarray:
    """Sum accumulated amounts without turning missing records into zero flux.

    Inactive raster cells may stay NaN. The exchange operator separately checks
    finiteness at the coupled row/column positions, where data are required.
    """
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return np.sum(array, axis=0, dtype=np.float64)
    raise error_type(
        f"{variable_name} has unsupported shape {array.shape} in {path}; expected (y,x) or (time,y,x)"
    )
