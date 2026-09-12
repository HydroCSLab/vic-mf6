"""Build a conservative VIC--MODFLOW 6 overlap table from model inputs.

The coupled time loop deliberately never performs geometry.  This module is the
slow, inspectable preprocessing layer: it discovers the active VIC image grid
from a VIC global/domain file, discovers GWF cell polygons from an MF6
simulation, computes exact horizontal intersections, validates coverage, and
writes the immutable CSV consumed by :class:`vicmf6.exchange.ExchangeTable`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..exchange import ExchangeTable


class ExchangeBuildError(RuntimeError):
    """Raised when preprocessing cannot construct a trustworthy exchange table."""


@dataclass(frozen=True, slots=True)
class VicSourceCell:
    vic_id: str
    row: int
    col: int
    area_m2: float
    latitude: float
    longitude: float
    polygon_lonlat: Any
    interface_elevation_m: float | None


@dataclass(frozen=True, slots=True)
class Mf6SourceCell:
    model: str
    node: int  # one-based local node number
    area_m2: float
    polygon: Any
    top_m: float | None
    bottom_m: float | None


@dataclass(frozen=True, slots=True)
class BuildArtifacts:
    exchange_table: Path
    summary_json: Path
    summary_text: Path
    overlap_rows: int
    vic_cells: int
    mf6_cells: int


def _optional_imports():
    try:
        import flopy
        from netCDF4 import Dataset
        from pyproj import CRS, Transformer
        from shapely.geometry import Polygon
        from shapely.ops import transform
        from shapely.strtree import STRtree
    except ImportError as exc:
        raise ExchangeBuildError(
            "exchange-table preprocessing requires vicmf6[preprocess] "
            "(netCDF4, FloPy, pyproj, and shapely)"
        ) from exc
    return Dataset, flopy, CRS, Transformer, Polygon, transform, STRtree


def parse_vic_global(path: str | Path) -> dict[str, Any]:
    """Parse the small subset of VIC global-file directives needed here."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ExchangeBuildError(f"VIC global file was not found: {source}")

    result: dict[str, Any] = {"DOMAIN_TYPE": {}}
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        key = fields[0].upper()
        if key == "DOMAIN_TYPE" and len(fields) >= 3:
            result["DOMAIN_TYPE"][fields[1].upper()] = fields[2]
        elif key in {"DOMAIN", "PARAMETERS"} and len(fields) >= 2:
            result[key] = _resolve_declared_path(fields[1], source.parent)

    if "DOMAIN" not in result:
        raise ExchangeBuildError(f"VIC global file does not declare DOMAIN: {source}")
    return result


def _resolve_declared_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ExchangeBuildError(
            f"unresolved environment variable in VIC path: {value!r}"
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _coordinate_edges(values: np.ndarray, label: str) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64).reshape(-1)
    if centers.size < 2 or not np.all(np.isfinite(centers)):
        raise ExchangeBuildError(f"{label} must contain at least two finite centers")
    delta = np.diff(centers)
    if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
        raise ExchangeBuildError(f"{label} centers must be strictly monotonic")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * delta[0]
    edges[-1] = centers[-1] + 0.5 * delta[-1]
    return edges


def _as_rectilinear_coordinates(
    lat: np.ndarray, lon: np.ndarray, shape: tuple[int, int]
):
    """Return 1-D row latitudes and column longitudes for a rectilinear VIC grid."""

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        if (lat.size, lon.size) != shape:
            raise ExchangeBuildError(
                f"VIC coordinate lengths {(lat.size, lon.size)} do not match domain shape {shape}"
            )
        return lat, lon
    if lat.shape == shape and lon.shape == shape:
        row_lat = lat[:, 0]
        col_lon = lon[0, :]
        if not np.allclose(lat, row_lat[:, None], rtol=0.0, atol=1.0e-12):
            raise ExchangeBuildError(
                "curvilinear VIC latitude grid is not yet supported; provide a rectilinear domain"
            )
        if not np.allclose(lon, col_lon[None, :], rtol=0.0, atol=1.0e-12):
            raise ExchangeBuildError(
                "curvilinear VIC longitude grid is not yet supported; provide a rectilinear domain"
            )
        return row_lat, col_lon
    raise ExchangeBuildError(
        f"unsupported VIC LAT/LON shapes lat={lat.shape} lon={lon.shape} domain={shape}"
    )


def load_vic_cells(
    global_file: str | Path,
    *,
    interface_elevation_m: float | None = None,
) -> tuple[list[VicSourceCell], dict[str, Any]]:
    Dataset, _flopy, _CRS, _Transformer, Polygon, _transform, _STRtree = (
        _optional_imports()
    )
    global_path = Path(global_file).expanduser().resolve()
    parsed = parse_vic_global(global_path)
    domain_path = Path(parsed["DOMAIN"])
    if not domain_path.is_file():
        raise ExchangeBuildError(
            f"VIC DOMAIN declared by {global_path} was not found: {domain_path}"
        )

    names = parsed["DOMAIN_TYPE"]
    lat_name = names.get("LAT", "lat")
    lon_name = names.get("LON", "lon")
    mask_name = names.get("MASK", "mask")
    area_name = names.get("AREA", "area")
    frac_name = names.get("FRAC", "frac")

    with Dataset(domain_path, "r") as dataset:
        for required in (lat_name, lon_name, mask_name):
            if required not in dataset.variables:
                raise ExchangeBuildError(
                    f"VIC domain {domain_path} is missing variable {required!r}"
                )
        lat = np.asarray(dataset.variables[lat_name][:], dtype=np.float64)
        lon = np.asarray(dataset.variables[lon_name][:], dtype=np.float64)
        mask = np.asarray(dataset.variables[mask_name][:])
        if mask.ndim != 2:
            raise ExchangeBuildError(f"VIC mask must be 2-D, got shape {mask.shape}")
        area = (
            np.asarray(dataset.variables[area_name][:], dtype=np.float64)
            if area_name in dataset.variables
            else None
        )
        frac = (
            np.asarray(dataset.variables[frac_name][:], dtype=np.float64)
            if frac_name in dataset.variables
            else None
        )

    row_lat, col_lon = _as_rectilinear_coordinates(lat, lon, mask.shape)
    lat_edges = _coordinate_edges(row_lat, "VIC latitude")
    lon_edges = _coordinate_edges(col_lon, "VIC longitude")

    active = np.asarray(mask > 0, dtype=bool)
    if frac is not None:
        if frac.shape != mask.shape:
            raise ExchangeBuildError(
                f"VIC frac shape {frac.shape} does not match mask shape {mask.shape}"
            )
        active &= np.isfinite(frac) & (frac > 0.0)
    active_index = np.argwhere(active)
    if active_index.size == 0:
        raise ExchangeBuildError(f"VIC domain contains no active cells: {domain_path}")

    if area is not None and area.shape != mask.shape:
        raise ExchangeBuildError(
            f"VIC area shape {area.shape} does not match mask shape {mask.shape}"
        )

    # Geodesic area is used only when the domain file does not provide VIC AREA.
    geod = None
    if area is None:
        try:
            from pyproj import Geod
        except ImportError as exc:  # pragma: no cover - protected by optional import
            raise ExchangeBuildError(
                "pyproj is required to calculate VIC areas"
            ) from exc
        geod = Geod(ellps="WGS84")

    cells: list[VicSourceCell] = []
    for vic_position, (row, col) in enumerate(active_index.tolist()):
        lon0, lon1 = sorted((float(lon_edges[col]), float(lon_edges[col + 1])))
        lat0, lat1 = sorted((float(lat_edges[row]), float(lat_edges[row + 1])))
        polygon = Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])
        if not polygon.is_valid or polygon.area <= 0.0:
            raise ExchangeBuildError(f"invalid VIC polygon at row={row} col={col}")
        if area is not None:
            area_m2 = float(area[row, col])
        else:
            lon_values, lat_values = polygon.exterior.xy
            signed_area, _ = geod.polygon_area_perimeter(lon_values, lat_values)
            area_m2 = abs(float(signed_area))
        if not np.isfinite(area_m2) or area_m2 <= 0.0:
            raise ExchangeBuildError(
                f"invalid VIC area at row={row} col={col}: {area_m2}"
            )
        cells.append(
            VicSourceCell(
                vic_id=str(vic_position),
                row=int(row),
                col=int(col),
                area_m2=area_m2,
                latitude=float(row_lat[row]),
                longitude=float(col_lon[col]),
                polygon_lonlat=polygon,
                interface_elevation_m=(
                    None
                    if interface_elevation_m is None
                    else float(interface_elevation_m)
                ),
            )
        )

    area_values = np.asarray([cell.area_m2 for cell in cells], dtype=np.float64)
    info = {
        "global_file": str(global_path),
        "domain_file": str(domain_path),
        "grid_crs": "EPSG:4326",
        "shape": list(mask.shape),
        "active_cells": len(cells),
        "latitude_spacing_deg": _spacing_stats(row_lat),
        "longitude_spacing_deg": _spacing_stats(col_lon),
        "cell_area_m2": _stats(area_values),
        "active_bbox_lonlat": [
            min(cell.polygon_lonlat.bounds[0] for cell in cells),
            min(cell.polygon_lonlat.bounds[1] for cell in cells),
            max(cell.polygon_lonlat.bounds[2] for cell in cells),
            max(cell.polygon_lonlat.bounds[3] for cell in cells),
        ],
        "interface_elevation_m": (
            None if interface_elevation_m is None else float(interface_elevation_m)
        ),
    }
    return cells, info


def _spacing_stats(values: np.ndarray) -> dict[str, float]:
    diff = np.abs(np.diff(np.asarray(values, dtype=np.float64).reshape(-1)))
    return _stats(diff)


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _mf6_discretization(gwf: Any) -> str:
    """Return the exact MF6 discretization package type.

    FloPy convenience attributes can be ambiguous: for a DISU model,
    ``gwf.dis`` may resolve to the loaded DISU package.  Query package
    types explicitly and test the most specific unstructured forms first.
    """

    get_package = getattr(gwf, "get_package", None)
    if callable(get_package):
        for name in ("disu", "disv", "dis"):
            package = get_package(name)
            if package is not None:
                return name.upper()

    # Fallback for lightweight test doubles or older FloPy-like objects.
    for name in ("disu", "disv", "dis"):
        package = getattr(gwf, name, None)
        if package is not None:
            package_type = type(package).__name__.lower()
            if name in package_type or package_type == "simplenamespace":
                return name.upper()

    raise ExchangeBuildError(f"GWF model {gwf.name} has no DIS/DISV/DISU package")


def _array_or_none(package: Any, name: str):
    value = getattr(package, name, None)
    if value is None:
        return None
    array = getattr(value, "array", None)
    if array is None:
        return None
    return np.asarray(array)


def _selected_surface_nodes(gwf: Any, grid_type: str) -> list[int]:
    """Choose the shallowest active node for every horizontal footprint."""

    if grid_type == "DIS":
        dis = gwf.dis
        idomain = _array_or_none(dis, "idomain")
        nlay = int(dis.nlay.array)
        nrow = int(dis.nrow.array)
        ncol = int(dis.ncol.array)
        if idomain is None:
            idomain = np.ones((nlay, nrow, ncol), dtype=int)
        result = []
        for row in range(nrow):
            for col in range(ncol):
                layers = np.flatnonzero(idomain[:, row, col] > 0)
                if layers.size:
                    layer = int(layers[0])
                    result.append(layer * nrow * ncol + row * ncol + col)
        return result

    if grid_type == "DISV":
        disv = gwf.disv
        idomain = _array_or_none(disv, "idomain")
        nlay = int(disv.nlay.array)
        ncpl = int(disv.ncpl.array)
        if idomain is None:
            idomain = np.ones((nlay, ncpl), dtype=int)
        result = []
        for icpl in range(ncpl):
            layers = np.flatnonzero(idomain[:, icpl] > 0)
            if layers.size:
                result.append(int(layers[0]) * ncpl + icpl)
        return result

    # DISU has no general layer index.  Geometry is used below to collapse
    # duplicate horizontal footprints to the node with the highest cell top.
    disu = gwf.disu
    idomain = _array_or_none(disu, "idomain")
    nodes = int(disu.nodes.array)
    if idomain is None:
        return list(range(nodes))
    return [int(index) for index in np.flatnonzero(idomain.reshape(-1) > 0)]


def _node_vertical_bounds(
    gwf: Any, grid_type: str, node_zero: int
) -> tuple[float, float | None]:
    package = getattr(gwf, grid_type.lower())
    if grid_type == "DIS":
        nrow = int(package.nrow.array)
        ncol = int(package.ncol.array)
        ncpl = nrow * ncol
        layer, local = divmod(int(node_zero), ncpl)
        top2d = np.asarray(package.top.array, dtype=np.float64).reshape(-1)
        botm = np.asarray(package.botm.array, dtype=np.float64).reshape(
            int(package.nlay.array), ncpl
        )
        top = float(top2d[local] if layer == 0 else botm[layer - 1, local])
        return top, float(botm[layer, local])
    if grid_type == "DISV":
        ncpl = int(package.ncpl.array)
        layer, local = divmod(int(node_zero), ncpl)
        top1d = np.asarray(package.top.array, dtype=np.float64).reshape(-1)
        botm = np.asarray(package.botm.array, dtype=np.float64).reshape(
            int(package.nlay.array), ncpl
        )
        top = float(top1d[local] if layer == 0 else botm[layer - 1, local])
        return top, float(botm[layer, local])
    top = np.asarray(package.top.array, dtype=np.float64).reshape(-1)
    bottom = np.asarray(package.bot.array, dtype=np.float64).reshape(-1)
    return float(top[node_zero]), float(bottom[node_zero])


def load_mf6_cells(
    simulation_namefile: str | Path,
    *,
    mf6_crs: str | None = None,
) -> tuple[list[Mf6SourceCell], dict[str, Any]]:
    _Dataset, flopy, CRS, _Transformer, Polygon, _transform, _STRtree = (
        _optional_imports()
    )
    sim_file = Path(simulation_namefile).expanduser().resolve()
    if not sim_file.is_file():
        raise ExchangeBuildError(f"MF6 simulation name file was not found: {sim_file}")
    if sim_file.name.lower() != "mfsim.nam":
        raise ExchangeBuildError(
            "--mf6-sim must point to the simulation name file mfsim.nam so all GWF models can be discovered"
        )

    try:
        simulation = flopy.mf6.MFSimulation.load(
            sim_ws=str(sim_file.parent),
            verbosity_level=0,
        )
    except Exception as exc:
        raise ExchangeBuildError(
            f"FloPy could not load MF6 simulation {sim_file}: {exc}"
        ) from exc

    cells: list[Mf6SourceCell] = []
    model_info: list[dict[str, Any]] = []
    common_crs = None
    for model_name in simulation.model_names:
        gwf = simulation.get_model(model_name)
        if gwf is None or str(getattr(gwf, "model_type", "")).lower() != "gwf6":
            continue
        grid_type = _mf6_discretization(gwf)
        grid = gwf.modelgrid
        grid_crs = (
            CRS.from_user_input(mf6_crs) if mf6_crs else getattr(grid, "crs", None)
        )
        if grid_crs is None:
            raise ExchangeBuildError(
                f"MF6 model {model_name} has no CRS metadata; supply --mf6-crs (for example EPSG:5070)"
            )
        grid_crs = CRS.from_user_input(grid_crs)
        if not grid_crs.is_projected:
            raise ExchangeBuildError(
                f"MF6 model {model_name} CRS must be projected in linear units, got {grid_crs.to_string()}"
            )
        axis = grid_crs.axis_info[0] if grid_crs.axis_info else None
        unit_factor = None if axis is None else axis.unit_conversion_factor
        if unit_factor is None or abs(float(unit_factor) - 1.0) > 1.0e-12:
            raise ExchangeBuildError(
                f"MF6 model {model_name} CRS must use metres because the coupling API is m3/day; "
                f"got unit {getattr(axis, 'unit_name', 'unknown')}"
            )
        if common_crs is None:
            common_crs = grid_crs
        elif common_crs != grid_crs:
            raise ExchangeBuildError(
                "all coupled MF6 GWF models must use the same horizontal CRS in this preprocessing version"
            )

        package = getattr(gwf, grid_type.lower())
        candidate_nodes = _selected_surface_nodes(gwf, grid_type)
        candidate: list[Mf6SourceCell] = []
        for node_zero in candidate_nodes:
            try:
                vertices = grid.get_cell_vertices(node=int(node_zero))
            except TypeError:
                vertices = grid.get_cell_vertices(int(node_zero))
            if vertices is None or len(vertices) < 3:
                raise ExchangeBuildError(
                    f"MF6 model {model_name} node {node_zero + 1} has no usable CELL2D/polygon geometry"
                )
            polygon = Polygon(vertices)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
                raise ExchangeBuildError(
                    f"MF6 model {model_name} node {node_zero + 1} has invalid polygon geometry"
                )
            top_value, bottom_value = _node_vertical_bounds(gwf, grid_type, node_zero)
            declared_area = None
            if grid_type == "DISU":
                area_array = _array_or_none(package, "area")
                if area_array is not None:
                    declared_area = float(np.asarray(area_array).reshape(-1)[node_zero])
            cell_area = float(polygon.area) if declared_area is None else declared_area
            if declared_area is not None:
                relative = abs(declared_area - float(polygon.area)) / max(
                    declared_area, 1.0
                )
                if relative > 1.0e-6:
                    raise ExchangeBuildError(
                        f"MF6 model {model_name} node {node_zero + 1} AREA differs from CELL2D polygon area "
                        f"by relative {relative:.3e}; grid geometry is internally inconsistent"
                    )
            candidate.append(
                Mf6SourceCell(
                    model=str(model_name).upper(),
                    node=int(node_zero) + 1,
                    area_m2=cell_area,
                    polygon=polygon,
                    top_m=top_value,
                    bottom_m=bottom_value,
                )
            )

        if grid_type == "DISU":
            # Collapse vertically stacked nodes sharing an identical horizontal
            # footprint.  The shallowest active control volume is the coupling
            # target.  Single-layer DISU models pass through unchanged.
            grouped: dict[bytes, Mf6SourceCell] = {}
            for cell in candidate:
                key = cell.polygon.normalize().wkb
                previous = grouped.get(key)
                if previous is None or (cell.top_m or -np.inf) > (
                    previous.top_m or -np.inf
                ):
                    grouped[key] = cell
            candidate = sorted(grouped.values(), key=lambda item: item.node)

        if not candidate:
            raise ExchangeBuildError(
                f"MF6 GWF model {model_name} has no active surface cells"
            )
        cells.extend(candidate)
        areas = [cell.area_m2 for cell in candidate]
        thickness = [
            cell.top_m - cell.bottom_m
            for cell in candidate
            if cell.top_m is not None and cell.bottom_m is not None
        ]
        model_info.append(
            {
                "model": str(model_name).upper(),
                "grid_type": grid_type,
                "coupled_surface_nodes": len(candidate),
                "node_min": min(cell.node for cell in candidate),
                "node_max": max(cell.node for cell in candidate),
                "cell_area_m2": _stats(areas),
                "top_m": _stats(
                    cell.top_m for cell in candidate if cell.top_m is not None
                ),
                "bottom_m": _stats(
                    cell.bottom_m for cell in candidate if cell.bottom_m is not None
                ),
                "thickness_m": _stats(thickness),
            }
        )

    if not cells or common_crs is None:
        raise ExchangeBuildError(f"MF6 simulation contains no GWF models: {sim_file}")
    info = {
        "simulation_namefile": str(sim_file),
        "grid_crs": common_crs.to_string(),
        "models": model_info,
        "coupled_surface_cells": len(cells),
        "cell_area_m2": _stats(cell.area_m2 for cell in cells),
        "bbox": [
            min(cell.polygon.bounds[0] for cell in cells),
            min(cell.polygon.bounds[1] for cell in cells),
            max(cell.polygon.bounds[2] for cell in cells),
            max(cell.polygon.bounds[3] for cell in cells),
        ],
    }
    return cells, info


def _coverage_stats(values: Iterable[float]) -> dict[str, float]:
    return _stats(values)


def build_exchange_table(
    *,
    vic_global: str | Path,
    mf6_sim: str | Path,
    output: str | Path,
    mf6_crs: str | None = None,
    interface_elevation_m: float | None = None,
    require_full_vic_coverage: bool = True,
    coverage_tolerance: float = 5.0e-4,
    force: bool = False,
) -> BuildArtifacts:
    _Dataset, _flopy, CRS, Transformer, _Polygon, transform, STRtree = (
        _optional_imports()
    )
    if coverage_tolerance < 0.0:
        raise ExchangeBuildError("coverage_tolerance must be nonnegative")
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not force:
        raise ExchangeBuildError(
            f"output already exists: {output_path}; pass --force to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vic_cells, vic_info = load_vic_cells(
        vic_global,
        interface_elevation_m=interface_elevation_m,
    )
    mf6_cells, mf6_info = load_mf6_cells(mf6_sim, mf6_crs=mf6_crs)
    target_crs = CRS.from_user_input(mf6_info["grid_crs"])
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    projected_vic = [
        transform(transformer.transform, cell.polygon_lonlat) for cell in vic_cells
    ]
    for cell, polygon in zip(vic_cells, projected_vic, strict=True):
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
            raise ExchangeBuildError(
                f"VIC cell {cell.vic_id} could not be projected into MF6 CRS {target_crs.to_string()}"
            )

    mf6_polygons = [cell.polygon for cell in mf6_cells]
    tree = STRtree(mf6_polygons)
    rows: list[dict[str, Any]] = []
    raw_coverage: list[float] = []
    overlap_counts_vic: list[int] = []
    per_mf6_overlap_count = np.zeros(len(mf6_cells), dtype=np.int64)
    per_mf6_coupled_area = np.zeros(len(mf6_cells), dtype=np.float64)

    for vic_cell, vic_polygon in zip(vic_cells, projected_vic, strict=True):
        indexes = np.asarray(
            tree.query(vic_polygon, predicate="intersects"), dtype=np.int64
        )
        local_rows: list[tuple[int, float]] = []
        raw_total = 0.0
        for index in indexes.tolist():
            intersection = vic_polygon.intersection(mf6_polygons[index])
            raw_area = float(intersection.area)
            if raw_area <= 0.0:
                continue
            raw_total += raw_area
            local_rows.append((index, raw_area))
        coverage_fraction = raw_total / float(vic_polygon.area)
        raw_coverage.append(coverage_fraction)
        overlap_counts_vic.append(len(local_rows))
        if (
            require_full_vic_coverage
            and abs(coverage_fraction - 1.0) > coverage_tolerance
        ):
            raise ExchangeBuildError(
                "MF6 domain does not fully cover active VIC cell "
                f"vic_id={vic_cell.vic_id} row={vic_cell.row} col={vic_cell.col}: "
                f"coverage={coverage_fraction:.12g}; allowed error={coverage_tolerance:.3e}"
            )
        if not local_rows:
            continue

        # Horizontal geometry determines fractional coverage.  Physical exchange
        # volume uses the authoritative VIC domain AREA, so scale projected
        # polygon areas to that declared area instead of silently substituting
        # projection-dependent polygon area.
        area_scale = (
            vic_cell.area_m2 / raw_total
            if require_full_vic_coverage
            else vic_cell.area_m2 / float(vic_polygon.area)
        )
        for index, raw_area in sorted(
            local_rows,
            key=lambda item: (mf6_cells[item[0]].model, mf6_cells[item[0]].node),
        ):
            mf6_cell = mf6_cells[index]
            overlap_area = raw_area * area_scale
            per_mf6_overlap_count[index] += 1
            per_mf6_coupled_area[index] += overlap_area
            rows.append(
                {
                    "vic_id": vic_cell.vic_id,
                    "vic_row": vic_cell.row,
                    "vic_col": vic_cell.col,
                    "vic_area_m2": vic_cell.area_m2,
                    "vic_lat": vic_cell.latitude,
                    "vic_lon": vic_cell.longitude,
                    "vic_interface_elevation_m": vic_cell.interface_elevation_m,
                    "mf6_model": mf6_cell.model,
                    "mf6_node": mf6_cell.node,
                    "mf6_area_m2": mf6_cell.area_m2,
                    "overlap_area_m2": overlap_area,
                }
            )

    if not rows:
        raise ExchangeBuildError(
            "VIC and MF6 domains do not overlap; no exchange table was written"
        )

    # Reuse the production runtime validator before committing the generated CSV.
    ExchangeTable.from_records(
        rows,
        require_full_vic_coverage=require_full_vic_coverage,
        coverage_relative_tolerance=max(coverage_tolerance, 1.0e-12),
    )

    fieldnames = [
        "vic_id",
        "vic_row",
        "vic_col",
        "vic_area_m2",
        "vic_lat",
        "vic_lon",
        "vic_interface_elevation_m",
        "mf6_model",
        "mf6_node",
        "mf6_area_m2",
        "overlap_area_m2",
    ]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row[key] is None else _csv_value(row[key])
                    for key in fieldnames
                }
            )
    temporary.replace(output_path)

    mf6_coverage = [
        float(per_mf6_coupled_area[index] / mf6_cells[index].area_m2)
        if mf6_cells[index].area_m2 > 0.0
        else 0.0
        for index in range(len(mf6_cells))
    ]
    summary = {
        "vic": vic_info,
        "mf6": mf6_info,
        "exchange": {
            "overlap_rows": len(rows),
            "total_exchange_area_m2": float(
                sum(float(row["overlap_area_m2"]) for row in rows)
            ),
            "vic_geometric_coverage_fraction": _coverage_stats(raw_coverage),
            "mf6_coupled_fraction": _coverage_stats(mf6_coverage),
            "overlaps_per_vic_cell": _stats(overlap_counts_vic),
            "overlaps_per_mf6_cell": _stats(per_mf6_overlap_count),
            "fully_covers_all_active_vic_cells": bool(
                all(abs(value - 1.0) <= coverage_tolerance for value in raw_coverage)
            ),
        },
        "output": str(output_path),
    }
    summary_json = output_path.with_suffix(".summary.json")
    summary_text = output_path.with_suffix(".summary.txt")
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    text = format_summary(summary)
    summary_text.write_text(text, encoding="utf-8")
    print(text, end="")
    return BuildArtifacts(
        exchange_table=output_path,
        summary_json=summary_json,
        summary_text=summary_text,
        overlap_rows=len(rows),
        vic_cells=len(vic_cells),
        mf6_cells=len(mf6_cells),
    )


def _csv_value(value: Any) -> str:
    if isinstance(value, float | np.floating):
        return f"{float(value):.17g}"
    return str(value)


def format_summary(summary: dict[str, Any]) -> str:
    vic = summary["vic"]
    mf6 = summary["mf6"]
    exchange = summary["exchange"]
    lines = [
        "VIC-MODFLOW 6 exchange-table build",
        "==================================",
        "",
        f"VIC global                : {vic['global_file']}",
        f"VIC domain                : {vic['domain_file']}",
        f"VIC grid shape            : {tuple(vic['shape'])}",
        f"VIC active cells          : {vic['active_cells']}",
        _stat_line("VIC cell area (m2)", vic["cell_area_m2"]),
        _stat_line("VIC dlat (deg)", vic["latitude_spacing_deg"]),
        _stat_line("VIC dlon (deg)", vic["longitude_spacing_deg"]),
        "",
        f"MF6 simulation            : {mf6['simulation_namefile']}",
        f"MF6 CRS                   : {mf6['grid_crs']}",
        f"MF6 coupled surface cells : {mf6['coupled_surface_cells']}",
        _stat_line("MF6 cell area (m2)", mf6["cell_area_m2"]),
    ]
    for model in mf6["models"]:
        lines.extend(
            [
                f"  model {model['model']}             : {model['grid_type']}, {model['coupled_surface_nodes']} coupled nodes",
                _stat_line("    thickness (m)", model["thickness_m"]),
            ]
        )
    lines.extend(
        [
            "",
            f"overlap rows              : {exchange['overlap_rows']}",
            f"total exchange area (m2)  : {exchange['total_exchange_area_m2']:.12g}",
            _stat_line(
                "VIC coverage fraction", exchange["vic_geometric_coverage_fraction"]
            ),
            _stat_line("MF6 coupled fraction", exchange["mf6_coupled_fraction"]),
            _stat_line("overlaps/VIC cell", exchange["overlaps_per_vic_cell"]),
            _stat_line("overlaps/MF6 cell", exchange["overlaps_per_mf6_cell"]),
            f"full active VIC coverage  : {exchange['fully_covers_all_active_vic_cells']}",
            "",
            f"[OK] exchange table       : {summary['output']}",
            f"[OK] summary JSON         : {Path(summary['output']).with_suffix('.summary.json')}",
            f"[OK] summary text         : {Path(summary['output']).with_suffix('.summary.txt')}",
            "",
        ]
    )
    return "\n".join(lines)


def _stat_line(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label:<27}: min {stats['min']:.12g}  "
        f"mean {stats['mean']:.12g}  max {stats['max']:.12g}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build and validate an immutable VIC-MF6 polygon-overlap table"
    )
    parser.add_argument(
        "--vic-global", required=True, help="VIC Image Driver global file"
    )
    parser.add_argument("--mf6-sim", required=True, help="MODFLOW 6 mfsim.nam")
    parser.add_argument("--output", required=True, help="exchange-table CSV to create")
    parser.add_argument(
        "--mf6-crs",
        help="override/define the MF6 horizontal CRS (for example EPSG:5070)",
    )
    parser.add_argument(
        "--interface-elevation-m",
        type=float,
        help="constant VIC soil-base elevation in the shared vertical datum",
    )
    parser.add_argument(
        "--allow-partial-vic-coverage",
        action="store_true",
        help="allow active VIC cells to be only partly covered by MF6",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=5.0e-4,
        help="absolute tolerance on geometric coverage fraction (default: 5e-4)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_exchange_table(
            vic_global=args.vic_global,
            mf6_sim=args.mf6_sim,
            output=args.output,
            mf6_crs=args.mf6_crs,
            interface_elevation_m=args.interface_elevation_m,
            require_full_vic_coverage=not args.allow_partial_vic_coverage,
            coverage_tolerance=args.coverage_tolerance,
            force=args.force,
        )
    except ExchangeBuildError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
