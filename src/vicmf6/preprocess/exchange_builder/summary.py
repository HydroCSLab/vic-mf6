"""Explain preprocessing geometry and coverage in machine and human summaries."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


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


def _coverage_stats(values: Iterable[float]) -> dict[str, float]:
    return _stats(values)


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
