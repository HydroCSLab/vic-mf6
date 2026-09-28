"""Read-only FloPy metadata access shared by groundwater output readers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError


def load_flopy_simulation(config: ApplicationConfig) -> Any:
    """load the original MF6 input deck with FloPy for metadata/output readers."""

    try:
        import flopy
    except ImportError as exc:
        raise PostprocessingError(
            "MODFLOW binary postprocessing requires FloPy; install vicmf6[post]"
        ) from exc

    try:
        return flopy.mf6.MFSimulation.load(
            sim_ws=str(config.mf6.workspace),
            verbosity_level=0,
        )
    except Exception as exc:
        raise PostprocessingError(
            f"FloPy could not load the MF6 simulation under {config.mf6.workspace}: {exc}"
        ) from exc


def _discover_output_path(model: Any, attribute: str, workspace: Path) -> Path | None:
    oc = getattr(model, "oc", None)
    if oc is None or not hasattr(oc, attribute):
        return None
    record = getattr(oc, attribute)
    try:
        data = record.get_data()
    except Exception:
        try:
            data = record.array
        except Exception:
            return None
    value = _first_string(data)
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _first_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.void) and value.dtype.names:
        for name in value.dtype.names:
            found = _first_string(value[name])
            if found:
                return found
        return None
    if isinstance(value, np.ndarray):
        for item in value.reshape(-1):
            found = _first_string(item)
            if found:
                return found
        return None
    if isinstance(value, tuple | list):
        for item in value:
            found = _first_string(item)
            if found:
                return found
    return None


def _model_array(
    model: Any, package_name: str, variable_name: str, size: int
) -> np.ndarray | None:
    package = getattr(model, package_name, None)
    if package is None:
        return None
    variable = getattr(package, variable_name, None)
    if variable is None:
        return None
    try:
        array = np.asarray(variable.array, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return array if array.size == size else None


def _first_model_array(
    model: Any, candidates: Iterable[tuple[str, str]], size: int
) -> np.ndarray | None:
    for package_name, variable_name in candidates:
        array = _model_array(model, package_name, variable_name, size)
        if array is not None:
            return array
    return None


def _top_bottom_from_modelgrid(
    grid: Any,
    size: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        top_botm = np.asarray(grid.top_botm, dtype=np.float64)
    except Exception:
        return None, None
    if top_botm.ndim < 2 or top_botm.shape[0] < 2:
        return None, None
    top = np.asarray(top_botm[:-1], dtype=np.float64).reshape(-1)
    bottom = np.asarray(top_botm[1:], dtype=np.float64).reshape(-1)
    if top.size != size or bottom.size != size:
        return None, None
    return top, bottom


def _grid_area(grid: Any, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(grid.area, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return array if array.size == size else None


def _time_step_length(times: np.ndarray, index: int) -> float:
    previous = 0.0 if index == 0 else float(times[index - 1])
    return float(times[index]) - previous
