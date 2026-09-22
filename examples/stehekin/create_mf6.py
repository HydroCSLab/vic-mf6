#!/usr/bin/env python3
"""Create the synthetic Stehekin MODFLOW 6 model with FloPy.

The model is a small spatial verification case for the VIC--MODFLOW 6
coupler.  Its input deck is generated in ``run/mf6`` at run time and is not a
source file that needs to be committed to the repository.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfapi,
    ModflowGwfdisu,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfsto,
    ModflowIms,
    ModflowTdis,
)

try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover - exercised by installation checks
    raise RuntimeError(
        "creating the spatial Stehekin model requires pyproj; "
        "install vicmf6[preprocess]"
    ) from exc

HERE = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = HERE / "run" / "mf6"
MODEL_NAME = "STEHEKIN"
MODEL_STEM = "stehekin"
MF6_CRS = "EPSG:5070"
LONGITUDE_EDGES_DEG = (-121.125, -121.0125, -120.8, -120.63125, -120.5)
LATITUDE_EDGES_DEG = (48.125, 48.26, 48.43, 48.625)
NROW = len(LATITUDE_EDGES_DEG) - 1
NCOL = len(LONGITUDE_EDGES_DEG) - 1
NODES = NROW * NCOL
EXPECTED_NJA = 46
AQUIFER_TOP_M = -1.0
AQUIFER_BOTTOM_M = -201.0
HYDRAULIC_CONDUCTIVITY_M_PER_DAY = 50.0
SPECIFIC_STORAGE_PER_M = 1.0e-5
SPECIFIC_YIELD = 0.1
SIMULATION_DAYS = 10.0
SIMULATION_STEPS = 10
LOW_INITIAL_HEAD_M = -112.0
HIGH_INITIAL_HEAD_M = -98.0
INITIAL_HEADS_M = tuple(
    LOW_INITIAL_HEAD_M if (row + col) % 2 == 0 else HIGH_INITIAL_HEAD_M
    for row in range(NROW)
    for col in range(NCOL)
)

REQUIRED_FILES = {
    "mfsim.nam",
    "time.tdis",
    "stehekin.ims",
    "stehekin.nam",
    "stehekin.disu",
    "stehekin.ic",
    "stehekin.npf",
    "stehekin.sto",
    "stehekin.api",
    "stehekin.oc",
}


def _polygon_area_centroid(points):
    twice_area = 0.0
    cx_num = 0.0
    cy_num = 0.0
    closed = points + [points[0]]
    for (x0, y0), (x1, y1) in zip(closed[:-1], closed[1:], strict=True):
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx_num += (x0 + x1) * cross
        cy_num += (y0 + y1) * cross
    signed_area = 0.5 * twice_area
    if signed_area == 0.0:
        raise RuntimeError("degenerate MF6 cell polygon")
    return abs(signed_area), (cx_num / (6.0 * signed_area), cy_num / (6.0 * signed_area))


def _distance_to_segment(point, start, end):
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        raise RuntimeError("zero-length MF6 shared edge")
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _geometry():
    transformer = Transformer.from_crs("EPSG:4326", MF6_CRS, always_xy=True)
    vertices = []
    vertex_id = {}
    for row, lat in enumerate(LATITUDE_EDGES_DEG):
        for col, lon in enumerate(LONGITUDE_EDGES_DEG):
            x, y = transformer.transform(lon, lat)
            identifier = len(vertices)
            vertex_id[row, col] = identifier
            vertices.append((identifier, float(x), float(y)))
    by_id = {identifier: (x, y) for identifier, x, y in vertices}
    cells = []
    for row in range(NROW):
        for col in range(NCOL):
            ids = [
                vertex_id[row, col],
                vertex_id[row + 1, col],
                vertex_id[row + 1, col + 1],
                vertex_id[row, col + 1],
            ]
            points = [by_id[index] for index in ids]
            area, centroid = _polygon_area_centroid(points)
            cells.append(dict(id=row * NCOL + col, row=row, col=col,
                              vertex_ids=ids, points=points, area=area,
                              centroid=centroid))
    return vertices, cells


def _connections(cells):
    by_rc = {(cell["row"], cell["col"]): cell for cell in cells}
    iac, ja, cl12, hwva, ihc = [], [], [], [], []
    for cell in cells:
        neighbors = []
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            other = by_rc.get((cell["row"] + dr, cell["col"] + dc))
            if other is not None:
                neighbors.append(other)
        neighbors.sort(key=lambda item: item["id"])
        row_ja = [cell["id"]]
        row_cl12 = [0.0]
        row_hwva = [0.0]
        row_ihc = [1]
        for other in neighbors:
            shared = list({tuple(point) for point in cell["points"]}
                          & {tuple(point) for point in other["points"]})
            if len(shared) != 2:
                raise RuntimeError("MF6 cells do not share exactly one edge")
            row_ja.append(other["id"])
            row_cl12.append(_distance_to_segment(cell["centroid"], shared[0], shared[1]))
            row_hwva.append(math.dist(shared[0], shared[1]))
            row_ihc.append(1)
        iac.append(len(row_ja))
        ja.extend(row_ja)
        cl12.extend(row_cl12)
        hwva.extend(row_hwva)
        ihc.extend(row_ihc)
    if sum(iac) != EXPECTED_NJA:
        raise RuntimeError(f"expected NJA={EXPECTED_NJA}, constructed {sum(iac)}")
    return iac, ja, ihc, cl12, hwva


def create_model(workspace: Path, *, force: bool = False) -> None:
    workspace = workspace.expanduser().resolve()
    if all((workspace / name).is_file() for name in REQUIRED_FILES) and not force:
        print(f"[OK] Stehekin MODFLOW 6 model already exists: {workspace}")
        return
    if workspace.exists() and any(workspace.iterdir()):
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    vertices, cells = _geometry()
    iac, ja, ihc, cl12, hwva = _connections(cells)
    simulation = MFSimulation(sim_name=MODEL_STEM, version="mf6", exe_name="mf6",
                              sim_ws=workspace, verbosity_level=0, write_headers=False)
    ModflowTdis(simulation, time_units="DAYS", nper=1,
                perioddata=[(SIMULATION_DAYS, SIMULATION_STEPS, 1.0)],
                filename="time.tdis")
    ims = ModflowIms(simulation, print_option="SUMMARY", complexity="MODERATE",
                     outer_dvclose=1.0e-12, outer_maximum=100,
                     inner_maximum=300, inner_dvclose=1.0e-12,
                     rcloserecord="1.0e-8 STRICT",
                     linear_acceleration="BICGSTAB", relaxation_factor=0.97,
                     filename="stehekin.ims")
    model = ModflowGwf(simulation, modelname=MODEL_NAME,
                       model_nam_file="stehekin.nam", save_flows=True)
    ModflowGwfdisu(
        model, length_units="METERS", nodes=NODES, nja=EXPECTED_NJA,
        nvert=len(vertices), top=AQUIFER_TOP_M, bot=AQUIFER_BOTTOM_M,
        area=[cell["area"] for cell in cells], idomain=1, iac=iac, ja=ja,
        ihc=ihc, cl12=cl12, hwva=hwva, vertices=vertices,
        cell2d=[(cell["id"], *cell["centroid"], len(cell["vertex_ids"]),
                 *cell["vertex_ids"]) for cell in cells], filename="stehekin.disu")
    ModflowGwfic(model, strt=list(INITIAL_HEADS_M), filename="stehekin.ic")
    ModflowGwfnpf(model, save_flows=True, icelltype=0,
                  k=HYDRAULIC_CONDUCTIVITY_M_PER_DAY, filename="stehekin.npf")
    ModflowGwfsto(model, save_flows=True, iconvert=0, ss=SPECIFIC_STORAGE_PER_M,
                  sy=SPECIFIC_YIELD, transient={0: True}, filename="stehekin.sto")
    ModflowGwfapi(model, save_flows=True, maxbound=NODES, filename="stehekin.api",
                  pname="VICAPI")
    ModflowGwfoc(model, head_filerecord="stehekin.hds",
                 budget_filerecord="stehekin.cbc", saverecord=[("HEAD", "ALL"),
                 ("BUDGET", "ALL")], filename="stehekin.oc")
    simulation.register_solution_package(ims, [model.name])
    simulation.write_simulation()
    print(f"[OK] created Stehekin MODFLOW 6 model with FloPy: {workspace}")
    print(f"[OK] model={MODEL_NAME} grid=DISU nodes={NODES} nja={EXPECTED_NJA}")
    print(f"[OK] horizontal CRS={MF6_CRS}; vertical range={AQUIFER_BOTTOM_M:g} to {AQUIFER_TOP_M:g} m")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    create_model(args.workspace, force=args.force)


if __name__ == "__main__":
    main()
