"""read completed-run diagnostics and model outputs without changing them."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ApplicationConfig
from ..errors import PostprocessingError
from ..exchange import ExchangeTable
from .types import Mf6Geometry, Mf6HeadSeries

_NUMERIC_WINDOW_FIELDS = {
    "step": int,
    "nonlinear_iterations_max": int,
}


def load_coupling_windows(path: Path) -> list[dict[str, Any]]:
    """read runtime window diagnostics and convert numeric fields eagerly."""

    if not path.is_file():
        raise PostprocessingError(f"coupling diagnostics were not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise PostprocessingError(f"coupling diagnostics contain no rows: {path}")

    converted: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=2):
        item: dict[str, Any] = {}
        for key, raw in row.items():
            if raw is None:
                item[key] = None
                continue
            text = raw.strip()
            if key in {"start", "end"}:
                item[key] = text
            elif text == "":
                item[key] = None
            elif key in _NUMERIC_WINDOW_FIELDS:
                try:
                    item[key] = _NUMERIC_WINDOW_FIELDS[key](float(text))
                except ValueError as exc:
                    raise PostprocessingError(
                        f"invalid {key} in {path} row {row_index}: {text!r}"
                    ) from exc
            else:
                try:
                    item[key] = float(text)
                except ValueError:
                    item[key] = text
        converted.append(item)
    return converted


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PostprocessingError(f"JSON output was not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PostprocessingError(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PostprocessingError(f"expected a JSON object in {path}")
    return value


def load_exchange_records(path: Path) -> list[dict[str, Any]]:
    """read the static overlap table for reporting and plotting metadata."""

    if not path.is_file():
        raise PostprocessingError(f"exchange table was not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise PostprocessingError(f"exchange table contains no overlap rows: {path}")

    integer_fields = {"vic_row", "vic_col", "mf6_node"}
    float_fields = {
        "vic_area_m2",
        "vic_lat",
        "vic_lon",
        "vic_interface_elevation_m",
        "mf6_area_m2",
        "overlap_area_m2",
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, raw in row.items():
            text = "" if raw is None else raw.strip()
            if key in integer_fields and text:
                item[key] = int(text)
            elif key in float_fields:
                item[key] = None if not text else float(text)
            else:
                item[key] = text
        result.append(item)
    return result


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


def load_mf6_head_series(
    config: ApplicationConfig,
    simulation: Any,
) -> dict[str, Mf6HeadSeries]:
    """read initial heads and all saved MF6 head records for every GWF model."""

    try:
        from flopy.utils import HeadFile, HeadUFile
    except ImportError as exc:
        raise PostprocessingError("FloPy head readers are unavailable") from exc

    result: dict[str, Mf6HeadSeries] = {}
    for source_model in config.mf6_source.models:
        model_name = source_model.name
        model = simulation.get_model(model_name)
        if model is None:
            model = simulation.get_model(model_name.lower())
        if model is None:
            raise PostprocessingError(f"FloPy did not load GWF model {model_name}")
        if getattr(model, "ic", None) is None:
            raise PostprocessingError(f"MF6 model {model_name} has no IC package")

        initial = np.asarray(model.ic.strt.array, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(initial)):
            raise PostprocessingError(
                f"initial heads contain non-finite values for {model_name}"
            )

        head_path = _discover_output_path(
            model, "head_filerecord", config.mf6.workspace
        )
        if head_path is None or not head_path.is_file():
            fallback = config.mf6.workspace / f"{model_name.lower()}.hds"
            head_path = fallback if fallback.is_file() else head_path
        if head_path is None or not head_path.is_file():
            raise PostprocessingError(
                f"saved MF6 head file was not found for {model_name}"
            )

        reader = None
        errors: list[str] = []
        for reader_class in (HeadFile, HeadUFile):
            try:
                reader = reader_class(str(head_path), precision="double")
                times = np.asarray(reader.get_times(), dtype=np.float64)
                if times.size:
                    break
            except Exception as exc:
                errors.append(f"{reader_class.__name__}: {exc}")
                reader = None
        if reader is None:
            raise PostprocessingError(
                f"failed to read MF6 heads {head_path}: {'; '.join(errors)}"
            )

        saved: list[np.ndarray] = []
        for time_value in times:
            array = np.asarray(
                reader.get_data(totim=float(time_value)), dtype=np.float64
            ).reshape(-1)
            if array.size != initial.size:
                raise PostprocessingError(
                    f"MF6 head record size mismatch for {model_name} at t={time_value}: "
                    f"expected={initial.size} actual={array.size}"
                )
            saved.append(array)
        heads = np.vstack([initial, *saved]) if saved else initial.reshape(1, -1)
        time_axis = np.concatenate(([0.0], times))
        result[model_name.upper()] = Mf6HeadSeries(
            model_name=model_name.upper(),
            times_days=time_axis,
            heads_m=heads,
            initial_heads_m=initial,
        )
    return result


def load_mf6_geometries(
    config: ApplicationConfig,
    simulation: Any,
    exchange_records: list[dict[str, Any]],
) -> dict[str, Mf6Geometry]:
    """discover cell area/elevation and the best available plotting geometry."""

    result: dict[str, Mf6Geometry] = {}
    for source_model in config.mf6_source.models:
        model_name = source_model.name.upper()
        model = simulation.get_model(source_model.name) or simulation.get_model(
            source_model.name.lower()
        )
        if model is None:
            raise PostprocessingError(
                f"FloPy did not load GWF model {source_model.name}"
            )

        grid = model.modelgrid
        node_count = int(getattr(grid, "nnodes", 0) or getattr(grid, "ncpl", 0) or 0)
        if node_count <= 0:
            initial = np.asarray(model.ic.strt.array).reshape(-1)
            node_count = int(initial.size)

        area = _model_array(model, "disu", "area", node_count)
        if area is None:
            area = _grid_area(grid, node_count)
        if area is None:
            area = _area_from_exchange(exchange_records, model_name, node_count)
        if area is None:
            area = np.full(node_count, np.nan, dtype=np.float64)

        top = _first_model_array(
            model, (("disu", "top"), ("dis", "top"), ("disv", "top")), node_count
        )
        bottom = _first_model_array(
            model, (("disu", "bot"), ("dis", "botm"), ("disv", "botm")), node_count
        )
        grid_top, grid_bottom = _top_bottom_from_modelgrid(grid, node_count)
        if top is None:
            top = grid_top
        if bottom is None:
            bottom = grid_bottom
        if top is None:
            top = np.full(node_count, np.nan, dtype=np.float64)
        if bottom is None:
            bottom = np.full(node_count, np.nan, dtype=np.float64)

        specific_storage = _model_array(model, "sto", "ss", node_count)
        specific_yield = _model_array(model, "sto", "sy", node_count)
        iconvert_float = _model_array(model, "sto", "iconvert", node_count)
        iconvert = (
            None
            if iconvert_float is None
            else np.asarray(iconvert_float, dtype=np.int64)
        )

        x, y, row, col, vertices = _geometry_from_modelgrid(grid, node_count)
        geometry_source = "mf6_modelgrid"

        if x is None or y is None:
            fixture = _load_adjacent_fixture_geometry(config, model_name, node_count)
            if fixture is not None:
                x, y, row, col, fixture_area = fixture
                if np.any(~np.isfinite(area)):
                    area = fixture_area
                geometry_source = "adjacent_reference_csv"
            else:
                x, y = _coupling_centroids(exchange_records, model_name, node_count)
                geometry_source = "overlap_weighted_vic_centroids"

        result[model_name] = Mf6Geometry(
            model_name=model_name,
            node=np.arange(1, node_count + 1, dtype=np.int64),
            area_m2=np.asarray(area, dtype=np.float64).reshape(-1),
            top_m=np.asarray(top, dtype=np.float64).reshape(-1),
            bottom_m=np.asarray(bottom, dtype=np.float64).reshape(-1),
            specific_storage_per_m=None
            if specific_storage is None
            else np.asarray(specific_storage, dtype=np.float64).reshape(-1),
            specific_yield=None
            if specific_yield is None
            else np.asarray(specific_yield, dtype=np.float64).reshape(-1),
            iconvert=iconvert,
            x=None if x is None else np.asarray(x, dtype=np.float64).reshape(-1),
            y=None if y is None else np.asarray(y, dtype=np.float64).reshape(-1),
            row=None if row is None else np.asarray(row, dtype=np.int64).reshape(-1),
            col=None if col is None else np.asarray(col, dtype=np.int64).reshape(-1),
            vertices=vertices,
            source=geometry_source,
        )
    return result


def load_mf6_budget_data(
    config: ApplicationConfig,
    simulation: Any,
    geometries: dict[str, Mf6Geometry],
) -> dict[str, dict[str, Any]]:
    """read CBC terms, API application, and FLOW-JA-FACE when available."""

    try:
        from flopy.utils import CellBudgetFile
    except ImportError as exc:
        raise PostprocessingError("FloPy budget readers are unavailable") from exc

    results: dict[str, dict[str, Any]] = {}
    for source_model in config.mf6_source.models:
        model_name = source_model.name.upper()
        model = simulation.get_model(source_model.name) or simulation.get_model(
            source_model.name.lower()
        )
        cbc_path = (
            _discover_output_path(model, "budget_filerecord", config.mf6.workspace)
            if model is not None
            else None
        )
        if cbc_path is None or not cbc_path.is_file():
            fallback = config.mf6.workspace / f"{source_model.name.lower()}.cbc"
            cbc_path = fallback if fallback.is_file() else cbc_path
        if cbc_path is None or not cbc_path.is_file():
            candidates = sorted(config.mf6.workspace.glob("*.cbc"))
            if len(candidates) == 1:
                cbc_path = candidates[0]
        if not cbc_path.is_file():
            results[model_name] = {
                "available": False,
                "reason": "budget file not found",
            }
            continue

        try:
            cbc = CellBudgetFile(str(cbc_path), precision="double")
            times = np.asarray(cbc.get_times(), dtype=np.float64)
            names = tuple(
                _decode_budget_name(value) for value in cbc.get_unique_record_names()
            )
        except Exception as exc:
            results[model_name] = {
                "available": False,
                "reason": f"could not read {cbc_path}: {exc}",
            }
            continue

        node_count = int(geometries[model_name].node.size)
        api_name = _select_api_budget_name(names, source_model.api_package)
        flowja_name = _select_budget_name(names, "FLOW-JA-FACE")

        api_by_time: dict[float, np.ndarray] = {}
        if api_name is not None:
            for time_value in times:
                values = _budget_node_vector(
                    cbc, api_name, float(time_value), node_count
                )
                if values is not None:
                    api_by_time[float(time_value)] = values

        aggregate_terms: list[dict[str, Any]] = []
        for time_index, time_value in enumerate(times):
            dt = _time_step_length(times, time_index)
            for name in names:
                if name == flowja_name:
                    continue
                vector = _budget_node_vector(cbc, name, float(time_value), node_count)
                if vector is None:
                    continue
                positive = float(np.sum(vector[vector > 0.0], dtype=np.float64))
                negative = float(np.sum(vector[vector < 0.0], dtype=np.float64))
                aggregate_terms.append(
                    {
                        "model": model_name,
                        "time_days": float(time_value),
                        "dt_days": dt,
                        "term": name,
                        "positive_rate_m3_day": positive,
                        "negative_rate_m3_day": negative,
                        "net_rate_m3_day": float(np.sum(vector, dtype=np.float64)),
                        "positive_volume_m3": positive * dt,
                        "negative_volume_m3": negative * dt,
                        "net_volume_m3": float(np.sum(vector, dtype=np.float64)) * dt,
                    }
                )

        lateral = _load_flowja(cbc_path, cbc, flowja_name, times, node_count)
        results[model_name] = {
            "available": True,
            "path": cbc_path,
            "times_days": times,
            "record_names": names,
            "api_record_name": api_name,
            "api_by_time": api_by_time,
            "aggregate_terms": aggregate_terms,
            "lateral": lateral,
        }
    return results


def exchange_table_object(config: ApplicationConfig) -> ExchangeTable:
    return ExchangeTable.from_csv(
        config.coupling.exchange_table,
        require_full_vic_coverage=config.coupling.require_full_vic_coverage,
        coverage_relative_tolerance=config.coupling.coverage_relative_tolerance,
    )


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
    raise PostprocessingError(
        f"{variable_name} has unsupported shape {array.shape} in {path}; "
        "expected (y,x) or (time,y,x)"
    )


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


def _area_from_exchange(
    records: list[dict[str, Any]], model_name: str, size: int
) -> np.ndarray | None:
    area = np.full(size, np.nan, dtype=np.float64)
    found = False
    for row in records:
        if str(row["mf6_model"]).upper() != model_name:
            continue
        value = row.get("mf6_area_m2")
        if value is None:
            continue
        node = int(row["mf6_node"]) - 1
        if 0 <= node < size:
            area[node] = float(value)
            found = True
    return area if found else None


def _geometry_from_modelgrid(
    grid: Any,
    size: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    tuple[tuple[tuple[float, float], ...], ...] | None,
]:
    try:
        x = np.asarray(grid.xcellcenters, dtype=np.float64).reshape(-1)
        y = np.asarray(grid.ycellcenters, dtype=np.float64).reshape(-1)
        if (
            x.size != size
            or y.size != size
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
        ):
            x = y = None
    except Exception:
        x = y = None

    row = col = None
    try:
        nrow = int(grid.nrow)
        ncol = int(grid.ncol)
        nlay = int(getattr(grid, "nlay", 1))
        if nlay * nrow * ncol == size:
            row = np.tile(np.repeat(np.arange(nrow, dtype=np.int64), ncol), nlay)
            col = np.tile(np.arange(ncol, dtype=np.int64), nrow * nlay)
    except Exception:
        row = col = None

    vertices: list[tuple[tuple[float, float], ...]] = []
    if x is not None and y is not None:
        try:
            if row is not None and col is not None:
                nrow = int(grid.nrow)
                ncol = int(grid.ncol)
                for node in range(size):
                    within_layer = node % (nrow * ncol)
                    i = within_layer // ncol
                    j = within_layer % ncol
                    cell_vertices = grid.get_cell_vertices(i, j)
                    vertices.append(
                        tuple((float(px), float(py)) for px, py in cell_vertices)
                    )
            else:
                for node in range(size):
                    cell_vertices = grid.get_cell_vertices(node)
                    vertices.append(
                        tuple((float(px), float(py)) for px, py in cell_vertices)
                    )
        except Exception:
            vertices = []
    return x, y, row, col, tuple(vertices) if len(vertices) == size else None


def _load_adjacent_fixture_geometry(
    config: ApplicationConfig,
    model_name: str,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    path = config.coupling.exchange_table.parent / "reference" / "mf6_cells.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != node_count:
        return None
    try:
        ordered = sorted(rows, key=lambda row: int(row["mf6_id"]))
        ids = [int(row["mf6_id"]) for row in ordered]
        if ids != list(range(node_count)):
            return None
        x = np.array([float(row["center_lon"]) for row in ordered], dtype=np.float64)
        y = np.array([float(row["center_lat"]) for row in ordered], dtype=np.float64)
        r = np.array([int(row["row"]) for row in ordered], dtype=np.int64)
        c = np.array([int(row["col"]) for row in ordered], dtype=np.int64)
        area = np.array(
            [float(row["rectangle_area_m2"]) for row in ordered], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError):
        return None
    return x, y, r, c, area


def _coupling_centroids(
    records: list[dict[str, Any]],
    model_name: str,
    node_count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    x_sum = np.zeros(node_count, dtype=np.float64)
    y_sum = np.zeros(node_count, dtype=np.float64)
    area_sum = np.zeros(node_count, dtype=np.float64)
    for row in records:
        if str(row["mf6_model"]).upper() != model_name:
            continue
        lon = row.get("vic_lon")
        lat = row.get("vic_lat")
        if lon is None or lat is None:
            continue
        node = int(row["mf6_node"]) - 1
        area = float(row["overlap_area_m2"])
        x_sum[node] += float(lon) * area
        y_sum[node] += float(lat) * area
        area_sum[node] += area
    if not np.all(area_sum > 0.0):
        return None, None
    return x_sum / area_sum, y_sum / area_sum


def _decode_budget_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _select_budget_name(names: tuple[str, ...], target: str) -> str | None:
    target_key = target.strip().upper()
    for name in names:
        if name.upper() == target_key:
            return name
    return None


def _select_api_budget_name(names: tuple[str, ...], package_name: str) -> str | None:
    package_key = package_name.strip().upper()
    for name in names:
        if name.upper() == package_key:
            return name
    candidates = [name for name in names if "API" in name.upper()]
    return candidates[0] if len(candidates) == 1 else None


def _budget_node_vector(
    cbc: Any, text: str, totim: float, node_count: int
) -> np.ndarray | None:
    try:
        records = cbc.get_data(text=text, totim=totim)
    except Exception:
        return None
    if not records:
        return None
    total = np.zeros(node_count, dtype=np.float64)
    used = False
    for record in records:
        array = np.asarray(record)
        if array.dtype.names:
            names = {name.lower(): name for name in array.dtype.names}
            q_name = names.get("q")
            node_name = names.get("node") or names.get("node1")
            if q_name is None:
                continue
            q = np.asarray(array[q_name], dtype=np.float64).reshape(-1)
            if node_name is None:
                if q.size != node_count:
                    continue
                total += q
            else:
                nodes = np.asarray(array[node_name], dtype=np.int64).reshape(-1)
                if nodes.size != q.size:
                    continue
                zero = nodes - 1 if nodes.min(initial=1) >= 1 else nodes
                valid = (zero >= 0) & (zero < node_count)
                np.add.at(total, zero[valid], q[valid])
            used = True
            continue

        flat = np.asarray(array, dtype=np.float64).reshape(-1)
        if flat.size == node_count:
            total += flat
            used = True
    return total if used else None


def _time_step_length(times: np.ndarray, index: int) -> float:
    previous = 0.0 if index == 0 else float(times[index - 1])
    return float(times[index]) - previous


def _load_flowja(
    cbc_path: Path,
    cbc: Any,
    flowja_name: str | None,
    times: np.ndarray,
    node_count: int,
) -> dict[str, Any] | None:
    if flowja_name is None:
        return None
    grid_path = _find_grb(cbc_path.parent)
    if grid_path is None:
        return {"available": False, "reason": "binary grid file not found"}
    try:
        try:
            from flopy.mf6.utils.binarygrid_util import MfGrdFile
        except ImportError:
            from flopy.utils import MfGrdFile
        grb = MfGrdFile(str(grid_path))
        ia = np.asarray(grb.ia, dtype=np.int64).reshape(-1)
        ja = np.asarray(grb.ja, dtype=np.int64).reshape(-1)
    except Exception as exc:
        return {"available": False, "reason": f"could not read {grid_path}: {exc}"}

    if ia.size != node_count + 1:
        return {
            "available": False,
            "reason": f"IA length {ia.size} != nodes+1 {node_count + 1}",
        }
    if ia[0] == 1:
        ia = ia - 1
    if ja.min(initial=0) >= 1 and ja.max(initial=0) <= node_count:
        ja = ja - 1

    pairs: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for time_index, time_value in enumerate(times):
        try:
            raw_records = cbc.get_data(text=flowja_name, totim=float(time_value))
        except Exception:
            continue
        if not raw_records:
            continue
        flowja = np.asarray(raw_records[0], dtype=np.float64).reshape(-1)
        if flowja.size != ja.size:
            continue
        dt = _time_step_length(times, time_index)

        reverse: dict[tuple[int, int], float] = {}
        row_sum = np.zeros(node_count, dtype=np.float64)
        for i in range(node_count):
            for position in range(int(ia[i]), int(ia[i + 1])):
                j = int(ja[position])
                if i == j:
                    continue
                q = float(flowja[position])
                row_sum[i] += q
                reverse[(i, j)] = q

        for i in range(node_count):
            cells.append(
                {
                    "time_days": float(time_value),
                    "dt_days": dt,
                    "node": i + 1,
                    "flowja_row_net_rate_m3_day": float(row_sum[i]),
                    "flowja_row_net_volume_m3": float(row_sum[i]) * dt,
                }
            )
        for (i, j), q_ij in reverse.items():
            if i >= j:
                continue
            q_ji = reverse.get((j, i))
            if q_ji is None:
                continue
            pairs.append(
                {
                    "time_days": float(time_value),
                    "dt_days": dt,
                    "node_i": i + 1,
                    "node_j": j + 1,
                    "q_i_to_j_record_m3_day": q_ij,
                    "q_j_to_i_record_m3_day": q_ji,
                    "antisymmetry_error_m3_day": q_ij + q_ji,
                    "pair_magnitude_m3_day": 0.5 * (abs(q_ij) + abs(q_ji)),
                    "pair_volume_m3": 0.5 * (abs(q_ij) + abs(q_ji)) * dt,
                }
            )
    return {
        "available": bool(cells),
        "grid_path": grid_path,
        "cell_rows": cells,
        "pair_rows": pairs,
    }


def _find_grb(workspace: Path) -> Path | None:
    candidates = sorted(workspace.glob("*.grb"))
    return candidates[0] if len(candidates) == 1 else None
