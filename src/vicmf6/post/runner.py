"""orchestrate completed-run extraction, acceptance, reporting, and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ApplicationConfig
from ..errors import PostprocessingError
from .acceptance import evaluate_acceptance
from .figures import create_all_figures
from .loaders import (
    exchange_table_object,
    load_coupling_windows,
    load_exchange_records,
    load_flopy_simulation,
    load_json,
    load_mf6_budget_data,
    load_mf6_geometries,
    load_mf6_head_series,
    load_vic_exchange_fields,
)
from .metrics import (
    build_budget_term_table,
    build_cell_budget_table,
    build_lateral_tables,
    build_mapping_tables,
    build_mf6_head_tables,
    build_mf6_mass_balance_table,
    build_node_boundary_table,
    build_post_summary,
    build_vic_cell_table,
    build_vic_exchange_tables,
    enrich_mass_balance_summary,
    enrich_post_summary,
)
from .report import (
    make_post_paths,
    prepare_post_directories,
    write_acceptance,
    write_csv,
    write_json,
    write_markdown_report,
)

_TABLE_FIELDS: dict[str, list[str]] = {
    "vic_cells.csv": [
        "vic_position",
        "vic_id",
        "vic_row",
        "vic_col",
        "vic_area_m2",
        "vic_lat",
        "vic_lon",
        "vic_interface_elevation_m",
    ],
    "vic_mapping_summary.csv": [
        "vic_id",
        "vic_row",
        "vic_col",
        "vic_area_m2",
        "vic_lat",
        "vic_lon",
        "overlap_count",
        "coupled_area_m2",
        "coverage_fraction",
        "mf6_models",
        "mf6_nodes",
    ],
    "mf6_mapping_summary.csv": [
        "mf6_model",
        "mf6_node",
        "mf6_area_m2",
        "overlap_count",
        "coupled_area_m2",
        "coverage_fraction",
        "vic_cells",
    ],
    "mapping_summary.csv": ["metric", "value", "units"],
    "vic_exchange_daily.csv": [
        "step",
        "start",
        "end",
        "duration_days",
        "positive_m3",
        "negative_m3",
        "net_m3",
        "net_rate_m3_day",
        "maximum_water_error_mm",
        "source_files",
    ],
    "vic_exchange_cells.csv": [
        "step",
        "start",
        "end",
        "vic_position",
        "vic_id",
        "vic_row",
        "vic_col",
        "vic_lat",
        "vic_lon",
        "vic_area_m2",
        "exchange_mm",
        "exchange_volume_m3",
        "direction",
    ],
    "mf6_heads.csv": [
        "model",
        "time_days",
        "node",
        "head_m",
        "head_change_from_initial_m",
    ],
    "mf6_head_summary.csv": [
        "model",
        "time_days",
        "head_min_m",
        "head_mean_m",
        "head_max_m",
        "head_range_m",
    ],
    "mf6_cells.csv": [
        "model",
        "node",
        "area_m2",
        "top_m",
        "bottom_m",
        "x",
        "y",
        "row",
        "col",
        "initial_head_m",
        "final_head_m",
        "head_change_m",
        "geometry_source",
    ],
    "node_boundary_daily.csv": [
        "step",
        "start",
        "end",
        "model",
        "node",
        "target_volume_m3",
        "target_rate_m3_day",
        "applied_volume_m3",
        "applied_error_m3",
        "direction",
    ],
    "mf6_budget_terms.csv": [
        "model",
        "time_days",
        "dt_days",
        "term",
        "positive_rate_m3_day",
        "negative_rate_m3_day",
        "net_rate_m3_day",
        "positive_volume_m3",
        "negative_volume_m3",
        "net_volume_m3",
    ],
    "mf6_lateral_cells.csv": [
        "model",
        "time_days",
        "dt_days",
        "node",
        "flowja_row_net_rate_m3_day",
        "flowja_row_net_volume_m3",
    ],
    "mf6_lateral_pairs.csv": [
        "model",
        "time_days",
        "dt_days",
        "node_i",
        "node_j",
        "q_i_to_j_record_m3_day",
        "q_j_to_i_record_m3_day",
        "antisymmetry_error_m3_day",
        "pair_magnitude_m3_day",
        "pair_volume_m3",
    ],
    "mf6_cell_budget.csv": [
        "step",
        "start",
        "end",
        "model",
        "node",
        "head_before_m",
        "head_after_m",
        "storage_change_m3",
        "interface_volume_m3",
        "lateral_row_volume_m3",
        "expected_storage_change_m3",
        "cell_budget_residual_m3",
    ],
    "mf6_mass_balance.csv": [
        "model",
        "api_volume_m3",
        "storage_budget_volume_m3",
        "storage_state_change_m3",
        "other_nonstorage_volume_m3",
        "cbc_budget_residual_m3",
        "storage_terms",
        "other_nonstorage_terms",
        "complete",
        "reason",
    ],
}


def run_postprocessing(
    config: ApplicationConfig,
    *,
    action: str,
    output_directory: Path | None = None,
    figure_formats: tuple[str, ...] = ("png", "pdf"),
    figure_dpi: int = 220,
    strict_reference: bool = False,
) -> int:
    """run one postprocessing action without modifying raw model outputs."""

    # Backward-compatible no-op: archived H8c reference checking was removed.
    _ = strict_reference

    normalized = action.strip().lower()
    if normalized not in {"summarize", "figures", "accept", "all"}:
        raise PostprocessingError(f"unsupported postprocessing action: {action}")

    paths = make_post_paths(config.run_directory, output_directory)
    prepare_post_directories(paths)

    diagnostics_path = config.coupling.diagnostics_directory / "coupling_windows.csv"
    runtime_summary_path = config.coupling.diagnostics_directory / "run_summary.json"
    manifest_path = config.coupling.diagnostics_directory / "run_manifest.json"

    window_rows = load_coupling_windows(diagnostics_path)
    runtime_summary = load_json(runtime_summary_path)
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    exchange_records = load_exchange_records(config.coupling.exchange_table)
    exchange_table = exchange_table_object(config)
    vic_fields = load_vic_exchange_fields(config, expected_windows=len(window_rows))

    simulation = load_flopy_simulation(config)
    head_series = load_mf6_head_series(config, simulation)
    geometries = load_mf6_geometries(config, simulation, exchange_records)
    budget_data = load_mf6_budget_data(config, simulation, geometries)

    vic_cells = build_vic_cell_table(exchange_table)
    vic_mapping_rows, mf6_mapping_rows = build_mapping_tables(exchange_records)
    vic_exchange_cells, vic_exchange_summary = build_vic_exchange_tables(
        config,
        exchange_table,
        window_rows,
        vic_fields,
    )
    node_boundary_rows = build_node_boundary_table(
        config,
        exchange_table,
        window_rows,
        vic_fields,
        geometries,
        budget_data,
    )
    mf6_head_rows, mf6_head_summary, mf6_cells = build_mf6_head_tables(
        head_series,
        geometries,
    )
    budget_term_rows = build_budget_term_table(budget_data)
    lateral_cell_rows, lateral_pair_rows = build_lateral_tables(budget_data)
    mf6_mass_balance_rows = build_mf6_mass_balance_table(budget_data)
    cell_budget_rows = build_cell_budget_table(
        config,
        window_rows,
        head_series,
        geometries,
        node_boundary_rows,
        lateral_cell_rows,
    )

    summary = build_post_summary(
        config,
        window_rows,
        runtime_summary,
        vic_exchange_summary,
        mf6_head_summary,
        mf6_cells,
        node_boundary_rows,
        lateral_pair_rows,
        cell_budget_rows,
    )
    enrich_post_summary(summary, exchange_table=exchange_table)
    enrich_mass_balance_summary(summary, mf6_mass_balance_rows)
    summary["provenance"] = {
        "runtime_manifest": manifest,
        "config": str(config.source_path),
        "run_directory": str(config.run_directory),
    }
    summary["mf6_budget"] = _budget_availability_summary(budget_data)

    acceptance = evaluate_acceptance(
        config,
        window_rows,
        summary,
    )

    table_rows = {
        "coupling_windows.csv": window_rows,
        "exchange_overlaps.csv": exchange_records,
        "vic_cells.csv": vic_cells,
        "vic_mapping_summary.csv": vic_mapping_rows,
        "mf6_mapping_summary.csv": mf6_mapping_rows,
        "mapping_summary.csv": _mapping_global_rows(
            exchange_table,
            vic_mapping_rows,
            mf6_mapping_rows,
        ),
        "vic_exchange_daily.csv": vic_exchange_summary,
        "vic_exchange_cells.csv": vic_exchange_cells,
        "mf6_heads.csv": mf6_head_rows,
        "mf6_head_summary.csv": mf6_head_summary,
        "mf6_cells.csv": mf6_cells,
        "node_boundary_daily.csv": node_boundary_rows,
        "mf6_budget_terms.csv": budget_term_rows,
        "mf6_lateral_cells.csv": lateral_cell_rows,
        "mf6_lateral_pairs.csv": lateral_pair_rows,
        "mf6_cell_budget.csv": cell_budget_rows,
        "mf6_mass_balance.csv": mf6_mass_balance_rows,
    }
    for name, rows in table_rows.items():
        write_csv(paths.tables / name, rows, fieldnames=_TABLE_FIELDS.get(name))

    write_json(paths.summary_json, summary)
    write_acceptance(paths, acceptance)

    figure_names: list[str] = []
    if normalized in {"figures", "all"}:
        figure_names = create_all_figures(
            paths,
            window_rows=window_rows,
            vic_exchange_rows=vic_exchange_cells,
            vic_cell_rows=vic_cells,
            node_boundary_rows=node_boundary_rows,
            head_series=head_series,
            geometries=geometries,
            exchange_records=exchange_records,
            lateral_pair_rows=lateral_pair_rows,
            formats=figure_formats,
            dpi=figure_dpi,
        )

    notes = _post_notes(budget_data, geometries)
    write_markdown_report(
        paths,
        summary=summary,
        acceptance=acceptance,
        figure_names=figure_names,
        table_names=list(table_rows),
        notes=notes,
    )

    print(f"[OK] postprocessing outputs: {paths.root}")
    print(f"[{'PASS' if acceptance.passed else 'FAIL'}] numerical acceptance")
    print(f"[OK] tables: {len(table_rows)}")
    print(f"[OK] figures: {len(figure_names)}")
    print(f"[OK] report: {paths.report_markdown}")

    if not acceptance.passed:
        return 1
    return 0


def _mapping_global_rows(exchange_table, vic_rows, mf6_rows) -> list[dict[str, Any]]:
    return [
        {
            "metric": "vic_cells",
            "value": exchange_table.vic_cell_count,
            "units": "cells",
        },
        {
            "metric": "mf6_coupled_nodes",
            "value": len(mf6_rows),
            "units": "nodes",
        },
        {
            "metric": "overlap_rows",
            "value": exchange_table.overlap_count,
            "units": "records",
        },
        {
            "metric": "total_overlap_area",
            "value": exchange_table.total_overlap_area_m2,
            "units": "m2",
        },
        {
            "metric": "minimum_vic_coverage_fraction",
            "value": min(
                float(row["coverage_fraction"])
                for row in vic_rows
                if row["coverage_fraction"] is not None
            ),
            "units": "fraction",
        },
    ]


def _budget_availability_summary(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name, item in data.items():
        lateral = item.get("lateral") if item.get("available") else None
        result[model_name] = {
            "available": bool(item.get("available")),
            "path": str(item.get("path")) if item.get("path") is not None else None,
            "record_names": list(item.get("record_names", ())),
            "api_record_name": item.get("api_record_name"),
            "lateral_available": bool(lateral and lateral.get("available")),
            "reason": item.get("reason")
            or (lateral.get("reason") if lateral else None),
        }
    return result


def _post_notes(
    budget_data: dict[str, dict[str, Any]],
    geometries: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    for model_name, data in budget_data.items():
        if not data.get("available"):
            notes.append(
                f"{model_name}: MF6 budget data unavailable ({data.get('reason')})."
            )
            continue
        if data.get("api_record_name") is None:
            notes.append(
                f"{model_name}: API package record was not identifiable in the CBC file; node targets are reconstructed, but per-node saved API application is omitted."
            )
        lateral = data.get("lateral")
        if not lateral or not lateral.get("available"):
            reason = None if not lateral else lateral.get("reason")
            notes.append(
                f"{model_name}: FLOW-JA-FACE lateral postprocessing unavailable"
                + (f" ({reason})." if reason else ".")
            )
    for model_name, geometry in geometries.items():
        if geometry.source != "mf6_modelgrid":
            notes.append(
                f"{model_name}: plotting geometry source is {geometry.source}; numerical MF6 areas/heads remain model-derived where available."
            )
    return notes
