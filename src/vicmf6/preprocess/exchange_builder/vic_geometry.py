"""Build VIC source-cell polygons from model-owned domain and parameter files.

Coordinate order and active masks are checked before polygon intersection,
so the generated CSV preserves VIC row/column identity exactly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .dependencies import _optional_imports
from .summary import _spacing_stats, _stats
from .types import ExchangeBuildError, VicSourceCell


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
