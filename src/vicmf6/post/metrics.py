"""derive canonical tables and numerical metrics from completed-run outputs."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from ..config import ApplicationConfig
from ..errors import PostprocessingError
from ..exchange import ExchangeTable, SignedVolume
from .types import Mf6Geometry, Mf6HeadSeries


def build_vic_cell_table(
    exchange_table: ExchangeTable,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in exchange_table.vic_cells:
        rows.append(
            {
                "vic_position": cell.position,
                "vic_id": cell.vic_id,
                "vic_row": cell.row,
                "vic_col": cell.col,
                "vic_area_m2": cell.area_m2,
                "vic_lat": cell.latitude,
                "vic_lon": cell.longitude,
                "vic_interface_elevation_m": cell.interface_elevation_m,
            }
        )
    return rows


def build_mapping_tables(
    exchange_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """summarize static VIC and MF6 coupling coverage without losing identities."""

    vic: dict[tuple[str, int, int], dict[str, Any]] = {}
    mf6: dict[tuple[str, int], dict[str, Any]] = {}

    for row in exchange_records:
        vic_key = (str(row["vic_id"]), int(row["vic_row"]), int(row["vic_col"]))
        vic_item = vic.setdefault(
            vic_key,
            {
                "vic_id": vic_key[0],
                "vic_row": vic_key[1],
                "vic_col": vic_key[2],
                "vic_area_m2": row.get("vic_area_m2"),
                "vic_lat": row.get("vic_lat"),
                "vic_lon": row.get("vic_lon"),
                "overlap_count": 0,
                "coupled_area_m2": 0.0,
                "mf6_models": set(),
                "mf6_nodes": set(),
            },
        )
        vic_item["overlap_count"] += 1
        vic_item["coupled_area_m2"] += float(row["overlap_area_m2"])
        vic_item["mf6_models"].add(str(row["mf6_model"]).upper())
        vic_item["mf6_nodes"].add(
            f"{str(row['mf6_model']).upper()}:{int(row['mf6_node'])}"
        )

        mf6_key = (str(row["mf6_model"]).upper(), int(row["mf6_node"]))
        mf6_item = mf6.setdefault(
            mf6_key,
            {
                "mf6_model": mf6_key[0],
                "mf6_node": mf6_key[1],
                "mf6_area_m2": row.get("mf6_area_m2"),
                "overlap_count": 0,
                "coupled_area_m2": 0.0,
                "vic_cells": set(),
            },
        )
        mf6_item["overlap_count"] += 1
        mf6_item["coupled_area_m2"] += float(row["overlap_area_m2"])
        mf6_item["vic_cells"].add(f"{vic_key[0]}@{vic_key[1]},{vic_key[2]}")

    vic_rows: list[dict[str, Any]] = []
    for _, item in sorted(
        vic.items(), key=lambda pair: (pair[0][1], pair[0][2], pair[0][0])
    ):
        area = item["vic_area_m2"]
        item["coverage_fraction"] = (
            None
            if area in (None, 0.0)
            else float(item["coupled_area_m2"]) / float(area)
        )
        item["mf6_models"] = ";".join(sorted(item["mf6_models"]))
        item["mf6_nodes"] = ";".join(sorted(item["mf6_nodes"]))
        vic_rows.append(item)

    mf6_rows: list[dict[str, Any]] = []
    for _, item in sorted(mf6.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        area = item["mf6_area_m2"]
        item["coverage_fraction"] = (
            None
            if area in (None, 0.0)
            else float(item["coupled_area_m2"]) / float(area)
        )
        item["vic_cells"] = ";".join(sorted(item["vic_cells"]))
        mf6_rows.append(item)
    return vic_rows, mf6_rows


def build_vic_exchange_tables(
    config: ApplicationConfig,
    exchange_table: ExchangeTable,
    window_diagnostics: list[dict[str, Any]],
    vic_fields: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """build cell-level and window-level signed VIC exchange records."""

    if len(window_diagnostics) != len(vic_fields):
        raise PostprocessingError(
            "VIC output-window count does not match coupling diagnostics: "
            f"diagnostics={len(window_diagnostics)} outputs={len(vic_fields)}"
        )

    cell_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    areas = exchange_table.vic_area_vector_m2()

    for diagnostics, field_record in zip(window_diagnostics, vic_fields, strict=True):
        values_mm = exchange_table.extract_vic_values(field_record["field_mm"])
        volumes_m3 = values_mm * 1.0e-3 * areas
        signed = SignedVolume.from_values(volumes_m3)

        expected_net = float(diagnostics["vic_net_m3"])
        error = signed.net_m3 - expected_net
        allowed = config.coupling.conservation_absolute_tolerance_m3 + (
            config.coupling.conservation_relative_tolerance
            * max(abs(signed.net_m3), abs(expected_net))
        )
        if abs(error) > allowed:
            raise PostprocessingError(
                "reconstructed VIC exchange does not match runtime diagnostics: "
                f"step={diagnostics['step']} reconstructed={signed.net_m3:.17g} "
                f"diagnostic={expected_net:.17g} error={error:.6e} allowed={allowed:.6e}"
            )

        duration_days = _duration_days(diagnostics)
        summary_rows.append(
            {
                "step": int(diagnostics["step"]),
                "start": diagnostics["start"],
                "end": diagnostics["end"],
                "duration_days": duration_days,
                "positive_m3": signed.positive_m3,
                "negative_m3": signed.negative_m3,
                "net_m3": signed.net_m3,
                "net_rate_m3_day": signed.net_m3 / duration_days,
                "maximum_water_error_mm": field_record["water_error_max_mm"],
                "source_files": ";".join(str(path) for path in field_record["files"]),
            }
        )
        for cell, exchange_mm, volume_m3 in zip(
            exchange_table.vic_cells,
            values_mm,
            volumes_m3,
            strict=True,
        ):
            cell_rows.append(
                {
                    "step": int(diagnostics["step"]),
                    "start": diagnostics["start"],
                    "end": diagnostics["end"],
                    "vic_position": cell.position,
                    "vic_id": cell.vic_id,
                    "vic_row": cell.row,
                    "vic_col": cell.col,
                    "vic_lat": cell.latitude,
                    "vic_lon": cell.longitude,
                    "vic_area_m2": cell.area_m2,
                    "exchange_mm": float(exchange_mm),
                    "exchange_volume_m3": float(volume_m3),
                    "direction": _direction(float(volume_m3)),
                }
            )
    return cell_rows, summary_rows


def build_node_boundary_table(
    config: ApplicationConfig,
    exchange_table: ExchangeTable,
    window_diagnostics: list[dict[str, Any]],
    vic_fields: list[dict[str, Any]],
    geometries: dict[str, Mf6Geometry],
    budget_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """reconstruct node targets and compare them with saved API budget values."""

    rows: list[dict[str, Any]] = []
    for diagnostics, field_record in zip(window_diagnostics, vic_fields, strict=True):
        values_mm = exchange_table.extract_vic_values(field_record["field_mm"])
        window_start_day = _elapsed_days(config, diagnostics["start"])
        window_end_day = _elapsed_days(config, diagnostics["end"])
        duration_days = window_end_day - window_start_day

        for model_name in exchange_table.model_names:
            model_key = model_name.upper()
            node_count = int(geometries[model_key].node.size)
            mapping = exchange_table.map_vic_depth_to_model(
                model_name,
                values_mm,
                node_count=node_count,
            )
            target = np.asarray(mapping.volume_by_node_m3, dtype=np.float64)
            applied = _aggregate_api_volume(
                budget_data.get(model_key, {}),
                window_start_day,
                window_end_day,
                node_count,
            )
            for node_index in range(node_count):
                target_volume = float(target[node_index])
                applied_volume = None if applied is None else float(applied[node_index])
                rows.append(
                    {
                        "step": int(diagnostics["step"]),
                        "start": diagnostics["start"],
                        "end": diagnostics["end"],
                        "model": model_key,
                        "node": node_index + 1,
                        "target_volume_m3": target_volume,
                        "target_rate_m3_day": target_volume / duration_days,
                        "applied_volume_m3": applied_volume,
                        "applied_error_m3": (
                            None
                            if applied_volume is None
                            else applied_volume - target_volume
                        ),
                        "direction": _direction(target_volume),
                    }
                )
    return rows


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


def build_post_summary(
    config: ApplicationConfig,
    window_diagnostics: list[dict[str, Any]],
    runtime_summary: dict[str, Any],
    vic_summary_rows: list[dict[str, Any]],
    mf6_head_summary_rows: list[dict[str, Any]],
    mf6_cell_rows: list[dict[str, Any]],
    node_boundary_rows: list[dict[str, Any]],
    lateral_pair_rows: list[dict[str, Any]],
    cell_budget_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cumulative_vic = _signed_from_rows(
        vic_summary_rows, "positive_m3", "negative_m3", "net_m3"
    )
    targets = _signed_from_values(
        [float(row["target_volume_m3"]) for row in node_boundary_rows]
    )
    applied_values = [
        float(row["applied_volume_m3"])
        for row in node_boundary_rows
        if row["applied_volume_m3"] is not None
    ]
    applied = (
        _signed_from_values(applied_values)
        if node_boundary_rows and len(applied_values) == len(node_boundary_rows)
        else None
    )
    if applied is None:
        runtime_applied = (runtime_summary.get("cumulative") or {}).get("applied")
        if isinstance(runtime_applied, dict):
            applied = {
                "positive_m3": float(runtime_applied.get("positive_m3", 0.0)),
                "negative_m3": float(runtime_applied.get("negative_m3", 0.0)),
                "net_m3": float(runtime_applied.get("net_m3", 0.0)),
            }

    final_head_rows = [
        row
        for row in mf6_head_summary_rows
        if _is_final_head_row(row, mf6_head_summary_rows)
    ]
    initial_cells = [float(row["initial_head_m"]) for row in mf6_cell_rows]
    final_cells = [float(row["final_head_m"]) for row in mf6_cell_rows]
    changes = [float(row["head_change_m"]) for row in mf6_cell_rows]

    max_mapping = max(
        max(
            abs(float(row.get(name) or 0.0))
            for name in (
                "positive_mapping_error_m3",
                "negative_mapping_error_m3",
                "net_mapping_error_m3",
                "aggregation_net_error_m3",
            )
        )
        for row in window_diagnostics
    )
    max_api = max(
        max(
            abs(float(row.get(name) or 0.0))
            for name in (
                "positive_api_error_m3",
                "negative_api_error_m3",
                "net_api_error_m3",
            )
        )
        for row in window_diagnostics
    )
    water_values = [
        abs(float(row["maximum_vic_water_error_mm"]))
        for row in window_diagnostics
        if row.get("maximum_vic_water_error_mm") is not None
    ]

    return {
        "status": "completed" if len(window_diagnostics) else "unknown",
        "simulation": {
            "start": config.coupling.start_time.isoformat(),
            "end": config.coupling.end_time.isoformat(),
            "windows": len(window_diagnostics),
            "coupling_interval_days": config.coupling.interval_days,
            "mf6_models": [model.name.upper() for model in config.mf6_source.models],
        },
        "mapping": {},
        "exchange_m3": {
            "vic": cumulative_vic,
            "node_target": targets,
            "api_applied": applied,
            "runtime_cumulative": runtime_summary.get("cumulative"),
        },
        "heads_m": {
            "initial_min": min(initial_cells) if initial_cells else None,
            "initial_max": max(initial_cells) if initial_cells else None,
            "final_min": min(final_cells) if final_cells else None,
            "final_max": max(final_cells) if final_cells else None,
            "change_min": min(changes) if changes else None,
            "change_max": max(changes) if changes else None,
            "final_model_summaries": final_head_rows,
        },
        "cell_budget_m3": {
            "storage_change": (
                float(sum(float(row["storage_change_m3"]) for row in cell_budget_rows))
                if cell_budget_rows
                else None
            ),
            "interface": (
                float(
                    sum(float(row["interface_volume_m3"]) for row in cell_budget_rows)
                )
                if cell_budget_rows
                else None
            ),
            "lateral_row_sum": (
                float(
                    sum(float(row["lateral_row_volume_m3"]) for row in cell_budget_rows)
                )
                if cell_budget_rows
                else None
            ),
            "residual": (
                float(
                    sum(
                        float(row["cell_budget_residual_m3"])
                        for row in cell_budget_rows
                    )
                )
                if cell_budget_rows
                else None
            ),
        },
        "errors": {
            "maximum_mapping_or_aggregation_error_m3": max_mapping,
            "maximum_api_volume_error_m3": max_api,
            "maximum_vic_water_error_mm": max(water_values) if water_values else None,
            "maximum_lateral_pair_antisymmetry_m3_day": (
                max(
                    abs(float(row["antisymmetry_error_m3_day"]))
                    for row in lateral_pair_rows
                )
                if lateral_pair_rows
                else None
            ),
            "maximum_cell_budget_residual_m3": (
                max(
                    abs(float(row["cell_budget_residual_m3"]))
                    for row in cell_budget_rows
                )
                if cell_budget_rows
                else None
            ),
        },
        "timing_seconds": runtime_summary.get("timing_seconds"),
        "total_wall_seconds": runtime_summary.get("total_wall_seconds"),
    }


def enrich_post_summary(
    summary: dict[str, Any],
    *,
    exchange_table: ExchangeTable,
) -> dict[str, Any]:
    summary["mapping"] = {
        "vic_cells": exchange_table.vic_cell_count,
        "overlap_rows": exchange_table.overlap_count,
        "coupled_area_m2": exchange_table.total_overlap_area_m2,
        "mf6_models": list(exchange_table.model_names),
    }
    return summary


def cumulative_by_step(rows: list[dict[str, Any]], value_key: str) -> list[float]:
    total = 0.0
    values: list[float] = []
    for row in rows:
        total += float(row[value_key])
        values.append(total)
    return values


def _aggregate_api_volume(
    budget: dict[str, Any],
    start_day: float,
    end_day: float,
    node_count: int,
) -> np.ndarray | None:
    if not budget.get("available") or not budget.get("api_by_time"):
        return None
    times = np.asarray(budget.get("times_days", []), dtype=np.float64)
    api_by_time: dict[float, np.ndarray] = budget["api_by_time"]
    result = np.zeros(node_count, dtype=np.float64)
    used = False
    previous = 0.0
    for time_value in times:
        current = float(time_value)
        dt = current - previous
        if current > start_day + 1.0e-10 and current <= end_day + 1.0e-10:
            vector = api_by_time.get(current)
            if vector is not None:
                result += np.asarray(vector, dtype=np.float64) * dt
                used = True
        previous = current
    return result if used else None


def _head_at_time(series: Mf6HeadSeries, time_days: float) -> np.ndarray | None:
    difference = np.abs(series.times_days - float(time_days))
    index = int(np.argmin(difference))
    if difference[index] > 1.0e-8:
        return None
    return np.asarray(series.heads_m[index], dtype=np.float64)


def _duration_days(row: dict[str, Any]) -> float:
    start = datetime.fromisoformat(str(row["start"]))
    end = datetime.fromisoformat(str(row["end"]))
    duration = (end - start).total_seconds() / 86400.0
    if duration <= 0.0:
        raise PostprocessingError(
            f"non-positive coupling-window duration in step {row['step']}"
        )
    return duration


def _elapsed_days(config: ApplicationConfig, timestamp: str) -> float:
    value = datetime.fromisoformat(str(timestamp))
    return (value - config.coupling.start_time).total_seconds() / 86400.0


def _direction(value: float) -> str:
    if value > 0.0:
        return "vic_to_mf6"
    if value < 0.0:
        return "mf6_to_vic"
    return "zero"


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _signed_from_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "positive_m3": float(np.sum(array[array > 0.0], dtype=np.float64)),
        "negative_m3": float(np.sum(array[array < 0.0], dtype=np.float64)),
        "net_m3": float(np.sum(array, dtype=np.float64)),
    }


def _signed_from_rows(
    rows: list[dict[str, Any]],
    positive_key: str,
    negative_key: str,
    net_key: str,
) -> dict[str, float]:
    return {
        "positive_m3": float(sum(float(row[positive_key]) for row in rows)),
        "negative_m3": float(sum(float(row[negative_key]) for row in rows)),
        "net_m3": float(sum(float(row[net_key]) for row in rows)),
    }


def _is_final_head_row(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    model = row["model"]
    maximum = max(float(item["time_days"]) for item in rows if item["model"] == model)
    return np.isclose(float(row["time_days"]), maximum, atol=1.0e-10)


def build_mf6_mass_balance_table(
    budget_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-model and domain-integrated MF6 CBC mass-balance totals.

    MODFLOW CBC storage terms (for example STO-SS and STO-SY) are flows from
    storage into the groundwater equation.  The corresponding physical storage
    state change therefore has the opposite sign.
    """

    rows: list[dict[str, Any]] = []
    domain_api = 0.0
    domain_storage_budget = 0.0
    domain_other = 0.0
    complete = True
    reasons: list[str] = []

    for model_name, data in budget_data.items():
        if not data.get("available"):
            complete = False
            reasons.append(f"{model_name}: {data.get('reason') or 'CBC unavailable'}")
            continue

        api_name = data.get("api_record_name")
        if not api_name:
            complete = False
            reasons.append(f"{model_name}: API budget record not identifiable")
            continue

        by_term: dict[str, float] = defaultdict(float)
        for item in data.get("aggregate_terms", []):
            by_term[str(item["term"])] += float(item["net_volume_m3"])

        api_key = str(api_name).upper()
        api_volume = sum(
            value for term, value in by_term.items() if term.upper() == api_key
        )
        storage_terms = {
            term: value
            for term, value in by_term.items()
            if term.upper().startswith("STO-")
        }
        storage_budget = float(sum(storage_terms.values()))
        other_terms = {
            term: value
            for term, value in by_term.items()
            if term.upper() != api_key and not term.upper().startswith("STO-")
        }
        other_volume = float(sum(other_terms.values()))
        residual = api_volume + storage_budget + other_volume
        storage_state_change = -storage_budget

        rows.append(
            {
                "model": model_name,
                "api_volume_m3": api_volume,
                "storage_budget_volume_m3": storage_budget,
                "storage_state_change_m3": storage_state_change,
                "other_nonstorage_volume_m3": other_volume,
                "cbc_budget_residual_m3": residual,
                "storage_terms": ";".join(sorted(storage_terms)),
                "other_nonstorage_terms": ";".join(sorted(other_terms)),
                "complete": True,
                "reason": "",
            }
        )
        domain_api += api_volume
        domain_storage_budget += storage_budget
        domain_other += other_volume

    domain_residual = domain_api + domain_storage_budget + domain_other
    rows.append(
        {
            "model": "__DOMAIN__",
            "api_volume_m3": domain_api if complete else None,
            "storage_budget_volume_m3": domain_storage_budget if complete else None,
            "storage_state_change_m3": -domain_storage_budget if complete else None,
            "other_nonstorage_volume_m3": domain_other if complete else None,
            "cbc_budget_residual_m3": domain_residual if complete else None,
            "storage_terms": "",
            "other_nonstorage_terms": "",
            "complete": complete,
            "reason": "; ".join(reasons),
        }
    )
    return rows


def enrich_mass_balance_summary(
    summary: dict[str, Any],
    mf6_mass_balance_rows: list[dict[str, Any]],
) -> None:
    """Attach one headline end-to-end coupled mass balance to run_summary."""

    domain = next(
        (row for row in mf6_mass_balance_rows if row.get("model") == "__DOMAIN__"),
        None,
    )
    if domain is None or not domain.get("complete"):
        summary["mass_balance_m3"] = {
            "available": False,
            "reason": None if domain is None else domain.get("reason"),
        }
        return

    vic_net = float(summary["exchange_m3"]["vic"]["net_m3"])
    api_net = float(domain["api_volume_m3"])
    storage_budget = float(domain["storage_budget_volume_m3"])
    storage_state_change = float(domain["storage_state_change_m3"])
    other = float(domain["other_nonstorage_volume_m3"])
    cbc_residual = float(domain["cbc_budget_residual_m3"])

    cell_budget = summary.get("cell_budget_m3") or {}
    head_storage = cell_budget.get("storage_change")

    summary["mass_balance_m3"] = {
        "available": True,
        "vic_interface_net_m3": vic_net,
        "mf6_api_net_m3": api_net,
        "mf6_other_nonstorage_net_m3": other,
        "mf6_storage_budget_net_m3": storage_budget,
        "mf6_storage_state_change_m3": storage_state_change,
        "vic_to_api_residual_m3": api_net - vic_net,
        "mf6_cbc_residual_m3": cbc_residual,
        "end_to_end_residual_m3": storage_state_change - (vic_net + other),
        "head_derived_storage_change_m3": head_storage,
        "head_vs_cbc_storage_residual_m3": (
            None if head_storage is None else float(head_storage) - storage_state_change
        ),
    }
