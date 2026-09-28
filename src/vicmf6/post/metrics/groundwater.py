"""Calculate groundwater head, lateral-flow, and cell-storage diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ..types import Mf6Geometry, Mf6HeadSeries
from .time_series import _elapsed_days, _finite_or_none, _head_at_time


def build_mf6_head_tables(
    head_series: dict[str, Mf6HeadSeries],
    geometries: dict[str, Mf6Geometry],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """build complete head trajectories, summary statistics, and cell changes."""

    head_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for model_name in sorted(head_series):
        series = head_series[model_name]
        geometry = geometries[model_name]
        for time_index, time_days in enumerate(series.times_days):
            values = np.asarray(series.heads_m[time_index], dtype=np.float64)
            summary_rows.append(
                {
                    "model": model_name,
                    "time_days": float(time_days),
                    "head_min_m": float(np.min(values)),
                    "head_mean_m": float(np.mean(values)),
                    "head_max_m": float(np.max(values)),
                    "head_range_m": float(np.max(values) - np.min(values)),
                }
            )
            for node_index, value in enumerate(values):
                head_rows.append(
                    {
                        "model": model_name,
                        "time_days": float(time_days),
                        "node": node_index + 1,
                        "head_m": float(value),
                        "head_change_from_initial_m": float(
                            value - series.initial_heads_m[node_index]
                        ),
                    }
                )

        final = np.asarray(series.heads_m[-1], dtype=np.float64)
        for node_index in range(series.node_count):
            cell_rows.append(
                {
                    "model": model_name,
                    "node": node_index + 1,
                    "area_m2": _finite_or_none(geometry.area_m2[node_index]),
                    "top_m": _finite_or_none(geometry.top_m[node_index]),
                    "bottom_m": _finite_or_none(geometry.bottom_m[node_index]),
                    "x": None
                    if geometry.x is None
                    else _finite_or_none(geometry.x[node_index]),
                    "y": None
                    if geometry.y is None
                    else _finite_or_none(geometry.y[node_index]),
                    "row": None
                    if geometry.row is None
                    else int(geometry.row[node_index]),
                    "col": None
                    if geometry.col is None
                    else int(geometry.col[node_index]),
                    "initial_head_m": float(series.initial_heads_m[node_index]),
                    "final_head_m": float(final[node_index]),
                    "head_change_m": float(
                        final[node_index] - series.initial_heads_m[node_index]
                    ),
                    "geometry_source": geometry.source,
                }
            )
    return head_rows, summary_rows, cell_rows


def build_lateral_tables(
    budget_data: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for model_name, data in budget_data.items():
        lateral = data.get("lateral") if data.get("available") else None
        if not lateral or not lateral.get("available"):
            continue
        for row in lateral.get("cell_rows", []):
            cell_rows.append({"model": model_name, **row})
        for row in lateral.get("pair_rows", []):
            pair_rows.append({"model": model_name, **row})
    return cell_rows, pair_rows


def build_budget_term_table(
    budget_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in budget_data.values():
        if data.get("available"):
            rows.extend(data.get("aggregate_terms", []))
    return rows


def build_cell_budget_table(
    config: ApplicationConfig,
    window_diagnostics: list[dict[str, Any]],
    head_series: dict[str, Mf6HeadSeries],
    geometries: dict[str, Mf6Geometry],
    node_boundary_rows: list[dict[str, Any]],
    lateral_cell_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """evaluate connected confined-cell closure when storage metadata are sufficient."""

    target_lookup: dict[tuple[int, str, int], dict[str, Any]] = {
        (int(row["step"]), str(row["model"]), int(row["node"])): row
        for row in node_boundary_rows
    }
    lateral_lookup: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in lateral_cell_rows:
        lateral_lookup[(str(row["model"]), int(row["node"]))].append(
            (
                float(row["time_days"]),
                float(row["flowja_row_net_volume_m3"]),
            )
        )

    rows: list[dict[str, Any]] = []
    for model_name, series in head_series.items():
        geometry = geometries[model_name]
        if geometry.specific_storage_per_m is None or geometry.iconvert is None:
            continue
        if np.any(np.asarray(geometry.iconvert) != 0):
            continue
        thickness = geometry.top_m - geometry.bottom_m
        if (
            np.any(~np.isfinite(thickness))
            or np.any(thickness <= 0.0)
            or np.any(~np.isfinite(geometry.area_m2))
            or np.any(~np.isfinite(geometry.specific_storage_per_m))
        ):
            continue
        storage_factor = geometry.area_m2 * thickness * geometry.specific_storage_per_m

        for diagnostics in window_diagnostics:
            step = int(diagnostics["step"])
            end_day = _elapsed_days(config, diagnostics["end"])
            start_day = _elapsed_days(config, diagnostics["start"])
            head_before = _head_at_time(series, start_day)
            head_after = _head_at_time(series, end_day)
            if head_before is None or head_after is None:
                continue
            for node_index in range(series.node_count):
                target = target_lookup.get((step, model_name, node_index + 1))
                if target is None:
                    continue
                interface_volume = float(target["target_volume_m3"])
                lateral_volume = float(
                    sum(
                        volume
                        for time_value, volume in lateral_lookup.get(
                            (model_name, node_index + 1), []
                        )
                        if time_value > start_day + 1.0e-10
                        and time_value <= end_day + 1.0e-10
                    )
                )
                storage_change = float(
                    storage_factor[node_index]
                    * (head_after[node_index] - head_before[node_index])
                )
                expected = interface_volume + lateral_volume
                rows.append(
                    {
                        "step": step,
                        "start": diagnostics["start"],
                        "end": diagnostics["end"],
                        "model": model_name,
                        "node": node_index + 1,
                        "head_before_m": float(head_before[node_index]),
                        "head_after_m": float(head_after[node_index]),
                        "storage_change_m3": storage_change,
                        "interface_volume_m3": interface_volume,
                        "lateral_row_volume_m3": lateral_volume,
                        "expected_storage_change_m3": expected,
                        "cell_budget_residual_m3": storage_change - expected,
                    }
                )
    return rows
