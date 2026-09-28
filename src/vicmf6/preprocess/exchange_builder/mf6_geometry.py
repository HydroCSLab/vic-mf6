"""Extract coupled surface-cell geometry from DIS, DISV, or DISU models.

Model node numbers remain one-based in the exchange CSV. Surface selection
and vertical interface interpretation are decided here, not in the time loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dependencies import _optional_imports
from .summary import _stats
from .types import ExchangeBuildError, Mf6SourceCell


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
