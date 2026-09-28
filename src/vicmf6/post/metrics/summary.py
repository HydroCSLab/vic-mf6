"""Assemble report summaries from already reconstructed scientific tables."""

from __future__ import annotations

from typing import Any

from ...config import ApplicationConfig
from ...exchange import ExchangeTable
from .time_series import _is_final_head_row, _signed_from_rows, _signed_from_values


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
