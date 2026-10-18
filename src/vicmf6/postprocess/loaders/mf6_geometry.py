"""Resolve groundwater cell geometry for reports and spatial diagnostics.

Geometry discovery is kept outside the runtime. Optional plotting metadata
may be unavailable without changing the already executed coupling sequence."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ..records import Mf6Geometry
from .mf6_metadata import (
    _first_model_array,
    _grid_area,
    _model_array,
    _top_bottom_from_modelgrid,
)


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
            x, y = _coupling_centroids(exchange_records, model_name, node_count)
            geometry_source = (
                "overlap_weighted_vic_centroids" if x is not None else "unavailable"
            )

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
