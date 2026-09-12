#!/usr/bin/env python3

"""Build the synthetic, spatially explicit MODFLOW 6 Stehekin verification model.

This builder writes a complete one-layer GWF-DISU model under ``run/mf6``.
The groundwater model is intentionally synthetic: it exists to verify the
VIC--MODFLOW 6 coupling mechanics, not to represent calibrated Stehekin
hydrogeology.

The horizontal cell geometry is real geometry rather than a plotting sidecar.
The 12 groundwater cells are defined by four longitude intervals and three
latitude intervals covering the active Stehekin VIC example domain.  Their
vertices are projected to EPSG:5070 (CONUS Albers, metres) and written into the
DISU VERTICES/CELL2D blocks.  This lets FloPy and the exchange-table preprocessor
recover exact MF6 polygons directly from ``mfsim.nam``.

Vertical convention
-------------------
This verification case uses a local vertical datum because no surveyed surface
elevations are being represented:

* local land surface = 0 m;
* VIC soil-base / coupling interface = -1 m;
* MF6 cell top = -1 m;
* MF6 cell bottom = -201 m;
* confined verification thickness = 200 m.

The initial heads deliberately alternate between a lower and a higher value.
This creates a strong, inspectable hydraulic contrast and is intended to make
both coupling signs observable while MF6 also redistributes water laterally.
The values are numerical test inputs only.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "run" / "mf6"

MODEL_NAME = "STEHEKIN"
MODEL_STEM = "stehekin"
MF6_CRS = "EPSG:5070"

# These geographic edges reconstruct the nonmatching 3 x 4 groundwater grid
# used by the development verification case.  They intentionally do not align
# with the 1/8-degree VIC grid, so the exchange table contains many-to-many
# overlaps rather than trivial one-cell-to-one-cell mappings.
LONGITUDE_EDGES_DEG = (-121.125, -121.0125, -120.8, -120.63125, -120.5)
LATITUDE_EDGES_DEG = (48.125, 48.26, 48.43, 48.625)

NROW = len(LATITUDE_EDGES_DEG) - 1
NCOL = len(LONGITUDE_EDGES_DEG) - 1
NODES = NROW * NCOL
EXPECTED_NJA = 46

LOCAL_LAND_SURFACE_M = 0.0
VIC_SOIL_DEPTH_M = 1.0
VIC_INTERFACE_ELEVATION_M = LOCAL_LAND_SURFACE_M - VIC_SOIL_DEPTH_M
AQUIFER_TOP_M = VIC_INTERFACE_ELEVATION_M
AQUIFER_THICKNESS_M = 200.0
AQUIFER_BOTTOM_M = AQUIFER_TOP_M - AQUIFER_THICKNESS_M

HYDRAULIC_CONDUCTIVITY_M_PER_DAY = 50.0
SPECIFIC_STORAGE_PER_M = 1.0e-5
SPECIFIC_YIELD = 0.1  # ignored for ICONVERT=0, retained as valid STO metadata
SIMULATION_DAYS = 10.0
SIMULATION_STEPS = 10

# Deliberate checkerboard forcing around the hydraulic regime exercised by the
# VIC fixture.  Lower heads favor VIC->MF6 exchange; higher heads favor
# MF6->VIC support.  The exact realized sign still comes from VIC's bottom-layer
# hydraulic state, so postprocessing verifies that both signs actually occurred.
LOW_INITIAL_HEAD_M = -112.0
HIGH_INITIAL_HEAD_M = -98.0
INITIAL_HEADS_M = tuple(
    LOW_INITIAL_HEAD_M if (row + col) % 2 == 0 else HIGH_INITIAL_HEAD_M
    for row in range(NROW)
    for col in range(NCOL)
)


def _require_pyproj():
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            "building the spatial MF6 fixture requires pyproj; install vicmf6[preprocess]"
        ) from exc
    return Transformer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
        help="generated MF6 workspace (default: examples/stehekin/run/mf6)",
    )
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    geometry = _build_geometry()
    iac, ja_rows, cl12_rows, hwva_rows = _connections(geometry)
    nja = sum(iac)
    if nja != EXPECTED_NJA:
        raise RuntimeError(
            f"expected connected 3x4 topology NJA={EXPECTED_NJA}, constructed {nja}"
        )

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    _write(workspace, "mfsim.nam", _mfsim())
    _write(workspace, "time.tdis", _tdis())
    _write(workspace, f"{MODEL_STEM}.ims", _ims())
    _write(workspace, f"{MODEL_STEM}.nam", _gwf_name())
    _write(
        workspace,
        f"{MODEL_STEM}.disu",
        _disu(geometry, iac, ja_rows, cl12_rows, hwva_rows),
    )
    _write(workspace, f"{MODEL_STEM}.ic", _ic())
    _write(workspace, f"{MODEL_STEM}.npf", _npf())
    _write(workspace, f"{MODEL_STEM}.sto", _sto())
    _write(workspace, f"{MODEL_STEM}.api", _api())
    _write(workspace, f"{MODEL_STEM}.oc", _oc())

    areas = [cell["area"] for cell in geometry["cells"]]
    print(f"[OK] built Stehekin MODFLOW 6 model in {workspace}")
    print(f"[OK] model={MODEL_NAME} grid=DISU nodes={NODES} nja={nja}")
    print(f"[OK] horizontal CRS={MF6_CRS}")
    print(
        "[OK] vertical datum: "
        f"land={LOCAL_LAND_SURFACE_M:g} m "
        f"VIC/MF6 interface={VIC_INTERFACE_ELEVATION_M:g} m "
        f"MF6 bottom={AQUIFER_BOTTOM_M:g} m"
    )
    print(
        "[OK] cell area m2: "
        f"min={min(areas):.6g} mean={sum(areas) / len(areas):.6g} max={max(areas):.6g}"
    )
    print(
        "[OK] initial head pattern: "
        f"low={LOW_INITIAL_HEAD_M:g} m high={HIGH_INITIAL_HEAD_M:g} m"
    )
    print(
        "[OK] properties: "
        f"K={HYDRAULIC_CONDUCTIVITY_M_PER_DAY:g} m/day "
        f"Ss={SPECIFIC_STORAGE_PER_M:g} 1/m thickness={AQUIFER_THICKNESS_M:g} m"
    )
    print("[INFO] all groundwater values are numerical verification settings")


def _build_geometry() -> dict:
    """Project the 3 x 4 geographic grid to metric MF6 coordinates."""

    Transformer = _require_pyproj()
    transformer = Transformer.from_crs("EPSG:4326", MF6_CRS, always_xy=True)

    vertices: list[dict[str, float | int]] = []
    vertex_id: dict[tuple[int, int], int] = {}
    for row, lat in enumerate(LATITUDE_EDGES_DEG):
        for col, lon in enumerate(LONGITUDE_EDGES_DEG):
            x, y = transformer.transform(lon, lat)
            identifier = len(vertices) + 1
            vertex_id[(row, col)] = identifier
            vertices.append({"id": identifier, "x": float(x), "y": float(y)})

    by_id = {int(v["id"]): (float(v["x"]), float(v["y"])) for v in vertices}
    cells: list[dict] = []
    for row in range(NROW):
        for col in range(NCOL):
            # Clockwise order required by MODFLOW 6 CELL2D:
            # lower-left -> upper-left -> upper-right -> lower-right.
            ids = [
                vertex_id[(row, col)],
                vertex_id[(row + 1, col)],
                vertex_id[(row + 1, col + 1)],
                vertex_id[(row, col + 1)],
            ]
            points = [by_id[index] for index in ids]
            area, centroid = _polygon_area_centroid(points)
            cells.append(
                {
                    "id": row * NCOL + col,
                    "row": row,
                    "col": col,
                    "vertex_ids": ids,
                    "points": points,
                    "area": area,
                    "centroid": centroid,
                }
            )
    return {"vertices": vertices, "cells": cells}


def _polygon_area_centroid(points: list[tuple[float, float]]) -> tuple[float, tuple[float, float]]:
    """Return positive polygon area and centroid using the shoelace formula."""

    signed_twice_area = 0.0
    cx_numerator = 0.0
    cy_numerator = 0.0
    closed = points + [points[0]]
    for (x0, y0), (x1, y1) in zip(closed[:-1], closed[1:], strict=True):
        cross = x0 * y1 - x1 * y0
        signed_twice_area += cross
        cx_numerator += (x0 + x1) * cross
        cy_numerator += (y0 + y1) * cross
    signed_area = 0.5 * signed_twice_area
    if abs(signed_area) <= 0.0:
        raise RuntimeError("degenerate MF6 cell polygon")
    cx = cx_numerator / (6.0 * signed_area)
    cy = cy_numerator / (6.0 * signed_area)
    return abs(signed_area), (cx, cy)


def _distance_point_to_segment(point, start, end) -> float:
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 <= 0.0:
        raise RuntimeError("zero-length MF6 shared edge")
    t = ((px - x0) * dx + (py - y0) * dy) / length2
    t = max(0.0, min(1.0, t))
    qx = x0 + t * dx
    qy = y0 + t * dy
    return math.hypot(px - qx, py - qy)


def _shared_edge(cell: dict, other: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    first = {tuple(point) for point in cell["points"]}
    second = {tuple(point) for point in other["points"]}
    shared = list(first & second)
    if len(shared) != 2:
        raise RuntimeError(
            f"MF6 cells {cell['id']} and {other['id']} do not share exactly one edge"
        )
    return shared[0], shared[1]


def _connections(geometry: dict):
    """Build orthogonal DISU connectivity using actual projected cell geometry."""

    cells = geometry["cells"]
    by_rc = {(int(c["row"]), int(c["col"])): c for c in cells}
    iac: list[int] = []
    ja_rows: list[list[int]] = []
    cl12_rows: list[list[float]] = []
    hwva_rows: list[list[float]] = []

    for cell in cells:
        row = int(cell["row"])
        col = int(cell["col"])
        neighbors = []
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            other = by_rc.get((row + dr, col + dc))
            if other is not None:
                neighbors.append(other)
        neighbors.sort(key=lambda item: int(item["id"]))

        ja = [int(cell["id"]) + 1]
        cl12 = [0.0]
        hwva = [0.0]
        for other in neighbors:
            edge_start, edge_end = _shared_edge(cell, other)
            ja.append(int(other["id"]) + 1)
            cl12.append(
                _distance_point_to_segment(cell["centroid"], edge_start, edge_end)
            )
            hwva.append(math.dist(edge_start, edge_end))

        iac.append(len(ja))
        ja_rows.append(ja)
        cl12_rows.append(cl12)
        hwva_rows.append(hwva)

    return iac, ja_rows, cl12_rows, hwva_rows


def _header(description: str) -> str:
    return (
        f"# {description}\n"
        "# Generated by examples/stehekin/build_mf6.py.\n"
        "# Synthetic VIC-MODFLOW 6 verification model; not calibrated hydrogeology.\n\n"
    )


def _mfsim() -> str:
    return _header("MODFLOW 6 simulation name file") + f"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 time.tdis
END TIMING

BEGIN MODELS
  GWF6 {MODEL_STEM}.nam {MODEL_NAME}
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 {MODEL_STEM}.ims {MODEL_NAME}
END SOLUTIONGROUP
"""


def _tdis() -> str:
    return _header("Ten-day transient time discretization") + f"""BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  {SIMULATION_DAYS:.1f} {SIMULATION_STEPS} 1.0
END PERIODDATA
"""


def _ims() -> str:
    return _header("Iterative solver settings for the small verification model") + """BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY MODERATE
END OPTIONS

BEGIN NONLINEAR
  OUTER_DVCLOSE 1.0e-12
  OUTER_MAXIMUM 100
END NONLINEAR

BEGIN LINEAR
  INNER_MAXIMUM 300
  INNER_DVCLOSE 1.0e-12
  INNER_RCLOSE 1.0e-8 STRICT
  LINEAR_ACCELERATION BICGSTAB
  RELAXATION_FACTOR 0.97
END LINEAR
"""


def _gwf_name() -> str:
    return _header("Groundwater-flow model package list") + f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DISU6 {MODEL_STEM}.disu DISU
  IC6   {MODEL_STEM}.ic   IC
  NPF6  {MODEL_STEM}.npf  NPF
  STO6  {MODEL_STEM}.sto  STO
  API6  {MODEL_STEM}.api  VICAPI
  OC6   {MODEL_STEM}.oc   OC
END PACKAGES
"""


def _disu(geometry, iac, ja_rows, cl12_rows, hwva_rows) -> str:
    cells = geometry["cells"]
    vertices = geometry["vertices"]
    areas = " ".join(f"{float(cell['area']):.17g}" for cell in cells)
    iac_text = " ".join(str(value) for value in iac)
    ja_text = "\n".join("    " + " ".join(str(value) for value in row) for row in ja_rows)
    cl12_text = "\n".join(
        "    " + " ".join(f"{value:.17g}" for value in row) for row in cl12_rows
    )
    hwva_text = "\n".join(
        "    " + " ".join(f"{value:.17g}" for value in row) for row in hwva_rows
    )
    vertex_text = "\n".join(
        f"  {int(vertex['id'])} {float(vertex['x']):.17g} {float(vertex['y']):.17g}"
        for vertex in vertices
    )
    cell2d_text = "\n".join(
        "  "
        + " ".join(
            [
                str(int(cell["id"]) + 1),
                f"{float(cell['centroid'][0]):.17g}",
                f"{float(cell['centroid'][1]):.17g}",
                str(len(cell["vertex_ids"])),
                *(str(value) for value in cell["vertex_ids"]),
            ]
        )
        for cell in cells
    )

    return _header("Connected 12-node DISU groundwater grid with explicit polygons") + f"""BEGIN OPTIONS
  LENGTH_UNITS METERS
END OPTIONS

BEGIN DIMENSIONS
  NODES {NODES}
  NJA {EXPECTED_NJA}
  NVERT {len(vertices)}
END DIMENSIONS

BEGIN GRIDDATA
  TOP
    CONSTANT {AQUIFER_TOP_M:.17g}
  BOT
    CONSTANT {AQUIFER_BOTTOM_M:.17g}
  AREA
    INTERNAL FACTOR 1.0
    {areas}
  IDOMAIN
    CONSTANT 1
END GRIDDATA

BEGIN CONNECTIONDATA
  IHC
    CONSTANT 1
  IAC
    INTERNAL FACTOR 1
    {iac_text}
  JA
    INTERNAL FACTOR 1
{ja_text}
  CL12
    INTERNAL FACTOR 1.0
{cl12_text}
  HWVA
    INTERNAL FACTOR 1.0
{hwva_text}
END CONNECTIONDATA

BEGIN VERTICES
{vertex_text}
END VERTICES

BEGIN CELL2D
{cell2d_text}
END CELL2D
"""


def _ic() -> str:
    heads = " ".join(f"{value:.17g}" for value in INITIAL_HEADS_M)
    return _header("Deliberately heterogeneous bidirectional-test initial heads") + f"""BEGIN GRIDDATA
  STRT
    INTERNAL FACTOR 1.0
    {heads}
END GRIDDATA
"""


def _npf() -> str:
    return _header("Confined hydraulic properties and lateral groundwater flow") + f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICELLTYPE
    CONSTANT 0
  K
    CONSTANT {HYDRAULIC_CONDUCTIVITY_M_PER_DAY:.17g}
END GRIDDATA
"""


def _sto() -> str:
    return _header("Transient confined groundwater storage") + f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICONVERT
    CONSTANT 0
  SS
    CONSTANT {SPECIFIC_STORAGE_PER_M:.17g}
  SY
    CONSTANT {SPECIFIC_YIELD:.17g}
END GRIDDATA

BEGIN PERIOD 1
  TRANSIENT
END PERIOD
"""


def _api() -> str:
    return _header(
        "GWF-API source/sink populated at runtime by the VIC-MODFLOW 6 coupler"
    ) + f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND {NODES}
END DIMENSIONS
"""


def _oc() -> str:
    return _header("Head and cell-budget output required by postprocessing") + f"""BEGIN OPTIONS
  HEAD FILEOUT {MODEL_STEM}.hds
  BUDGET FILEOUT {MODEL_STEM}.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""


def _write(workspace: Path, name: str, text: str) -> None:
    (workspace / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
