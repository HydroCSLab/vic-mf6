"""Intersect model grids once and write a conservative overlap table.

All intersections use a common projected coordinate system. The resulting
areas must reproduce the declared VIC coverage before the table is accepted."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...exchange import ExchangeTable
from .dependencies import _optional_imports
from .mf6_geometry import load_mf6_cells
from .records import BuildArtifacts, ExchangeBuildError
from .summary import _coverage_stats, _stats, format_summary
from .vic_geometry import load_vic_cells


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
