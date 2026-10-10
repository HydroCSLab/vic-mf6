"""Calculate groundwater head, lateral-flow, and cell-storage diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ..records import Mf6Geometry, Mf6HeadSeries
from .time_series import _elapsed_days, _finite_or_none, _head_at_time


def build_mf6_head_summaries(
    head_series: dict[str, Mf6HeadSeries],
    geometries: dict[str, Mf6Geometry],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-time statistics and per-cell changes, reading one head record at a time."""
    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for model_name in sorted(head_series):
        series = head_series[model_name]
        geometry = geometries[model_name]
        for time_index, time_days in enumerate(series.times_days):
            values = np.asarray(series.read_heads(time_index), dtype=np.float64)
            values = values[np.isfinite(values)]
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

        final = np.asarray(series.read_heads(-1), dtype=np.float64)
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
                    "initial_head_m": _finite_or_none(
                        series.initial_heads_m[node_index]
                    ),
                    "final_head_m": _finite_or_none(final[node_index]),
                    "head_change_m": _finite_or_none(
                        final[node_index] - series.initial_heads_m[node_index]
                    ),
                    "geometry_source": geometry.source,
                }
            )
    return summary_rows, cell_rows


def iter_mf6_head_rows(head_series: dict[str, Mf6HeadSeries]):
    """Stream the node-by-time CSV without expanding all heads into dictionaries."""
    for model_name, series in sorted(head_series.items()):
        for time_index, time_days in enumerate(series.times_days):
            for node_index, value in enumerate(series.read_heads(time_index)):
                yield {
                    "model": model_name,
                    "time_days": float(time_days),
                    "node": node_index + 1,
                    "head_m": _finite_or_none(value),
                    "head_change_from_initial_m": _finite_or_none(
                        value - series.initial_heads_m[node_index]
                    ),
                }


def build_lateral_tables(
    budget_data: dict[str, dict[str, Any]],
    cell_rows,
    pair_rows,
) -> None:
    for model_name, data in budget_data.items():
        lateral = data.get("lateral") if data.get("available") else None
        if not lateral or not lateral.get("available"):
            continue
        for row in lateral.get("cell_rows", []):
            cell_rows.append({"model": model_name, **row})
        for row in lateral.get("pair_rows", []):
            pair_rows.append({"model": model_name, **row})


def build_budget_term_table(
    budget_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in budget_data.values():
        if data.get("available"):
            rows.extend(data.get("aggregate_terms", []))
    return rows


def iter_cell_budget_rows(
    config: ApplicationConfig,
    window_diagnostics: list[dict[str, Any]],
    head_series: dict[str, Mf6HeadSeries],
    geometries: dict[str, Mf6Geometry],
    node_boundary_rows: Iterable[dict[str, Any]],
    lateral_cell_rows: Iterable[dict[str, Any]],
    *,
    budget_data: dict[str, dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Evaluate the confined API-plus-lateral identity only where it is complete.

    Other stresses require their per-cell budgets; domain CBC closure already
    includes those terms. Missing FLOWJA evidence must never be treated as zero.
    """

    supported_models = {
        name
        for name, data in (budget_data or {}).items()
        if data.get("complete")
        and (data.get("lateral") or {}).get("available")
        and data.get("api_record_name")
        and all(
            term == data["api_record_name"]
            or term == "FLOW-JA-FACE"
            or term.startswith(("STO-", "DATA-"))
            for term in data["record_names"]
        )
    }
    if not supported_models:
        return

    for model_name, series in head_series.items():
        if model_name not in supported_models:
            continue
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

        # Tables are ordered by step/time within each model. Advance cursors
        # once instead of searching every saved record for every cell/window.
        targets = (row for row in node_boundary_rows if row["model"] == model_name)
        lateral = (row for row in lateral_cell_rows if row["model"] == model_name)
        target_row = next(targets, None)
        lateral_row = next(lateral, None)
        for diagnostics in window_diagnostics:
            step = int(diagnostics["step"])
            end_day = _elapsed_days(config, diagnostics["end"])
            start_day = _elapsed_days(config, diagnostics["start"])
            target_by_node = {}
            while target_row is not None and int(target_row["step"]) <= step:
                if int(target_row["step"]) == step:
                    target_by_node[int(target_row["node"])] = target_row
                target_row = next(targets, None)
            lateral_volumes = np.zeros(series.node_count, dtype=np.float64)
            while (
                lateral_row is not None
                and float(lateral_row["time_days"]) <= end_day + 1e-10
            ):
                if float(lateral_row["time_days"]) > start_day + 1e-10:
                    lateral_volumes[int(lateral_row["node"]) - 1] += float(
                        lateral_row["flowja_row_net_volume_m3"]
                    )
                lateral_row = next(lateral, None)
            head_before = _head_at_time(series, start_day)
            head_after = _head_at_time(series, end_day)
            if head_before is None or head_after is None:
                continue
            for node_index in range(series.node_count):
                target = target_by_node.get(node_index + 1)
                if target is None:
                    continue
                interface_volume = float(target["target_volume_m3"])
                lateral_volume = float(lateral_volumes[node_index])
                storage_change = float(
                    storage_factor[node_index]
                    * (head_after[node_index] - head_before[node_index])
                )
                expected = interface_volume + lateral_volume
                yield {
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
