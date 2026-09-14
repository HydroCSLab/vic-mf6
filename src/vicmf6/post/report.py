"""deterministic CSV/JSON/text/markdown writers for postprocessing products."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .acceptance import acceptance_as_dict, format_acceptance
from .types import AcceptanceResult, PostPaths


def make_post_paths(
    run_directory: Path, output_directory: Path | None = None
) -> PostPaths:
    root = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else run_directory / "postprocessing"
    )
    return PostPaths(
        root=root,
        tables=root / "tables",
        figures=root / "figures",
        report_markdown=root / "report.md",
        summary_json=root / "run_summary.json",
        acceptance_json=root / "acceptance.json",
        acceptance_text=root / "acceptance_summary.txt",
    )


def prepare_post_directories(paths: PostPaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str] | None = None
) -> None:
    items = list(rows)
    if fieldnames is None:
        fieldnames = _union_fieldnames(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in items:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_acceptance(paths: PostPaths, result: AcceptanceResult) -> None:
    write_json(paths.acceptance_json, acceptance_as_dict(result))
    paths.acceptance_text.write_text(format_acceptance(result), encoding="utf-8")


def write_markdown_report(
    paths: PostPaths,
    *,
    summary: dict[str, Any],
    acceptance: AcceptanceResult,
    figure_names: list[str],
    table_names: list[str],
    notes: list[str],
) -> None:
    exchange = summary.get("exchange_m3", {})
    vic = exchange.get("vic", {})
    target = exchange.get("node_target", {})
    heads = summary.get("heads_m", {})
    mass = summary.get("mass_balance_m3", {})
    errors = summary.get("errors", {})
    timing = summary.get("timing_seconds") or {}

    lines = [
        "# VIC–MODFLOW 6 coupled-run postprocessing report",
        "",
        "## Run status",
        "",
        f"- Numerical acceptance: **{'PASS' if acceptance.passed else 'FAIL'}**",
        f"- Coupling windows: {summary.get('simulation', {}).get('windows')}",
        f"- Simulation: `{summary.get('simulation', {}).get('start')}` to `{summary.get('simulation', {}).get('end')}`",
        "",
        "## Coupled exchange",
        "",
        f"- VIC gross positive: `{_fmt(vic.get('positive_m3'))} m³`",
        f"- VIC gross negative: `{_fmt(vic.get('negative_m3'))} m³`",
        f"- VIC net: `{_fmt(vic.get('net_m3'))} m³`",
        f"- MF6 node-target net: `{_fmt(target.get('net_m3'))} m³`",
        f"- Exchange directions observed: `{_exchange_direction(vic)}`",
        "",
        "## End-to-end mass balance",
        "",
        f"- MF6 API net (CBC): `{_fmt(mass.get('mf6_api_net_m3'))} m³`",
        f"- Other MF6 non-storage net: `{_fmt(mass.get('mf6_other_nonstorage_net_m3'))} m³`",
        f"- MF6 storage-state change: `{_fmt(mass.get('mf6_storage_state_change_m3'))} m³`",
        f"- VIC→MF6 API residual: `{_fmt(mass.get('vic_to_api_residual_m3'))} m³`",
        f"- **End-to-end residual:** `{_fmt(mass.get('end_to_end_residual_m3'))} m³`",
        f"- MF6 CBC domain residual: `{_fmt(mass.get('mf6_cbc_residual_m3'))} m³`",
        f"- Head-derived vs CBC storage residual: `{_fmt(mass.get('head_vs_cbc_storage_residual_m3'))} m³`",
        "",
        "The headline closure is `MF6 storage change = VIC interface transfer + other non-storage MF6 terms`. Internal FLOW-JA-FACE transfers cancel over the complete domain.",
        "",
        "## Groundwater heads",
        "",
        f"- Initial range: `{_fmt(heads.get('initial_min'))}` to `{_fmt(heads.get('initial_max'))} m`",
        f"- Final range: `{_fmt(heads.get('final_min'))}` to `{_fmt(heads.get('final_max'))} m`",
        f"- Head-change range: `{_fmt(heads.get('change_min'))}` to `{_fmt(heads.get('change_max'))} m`",
        "",
        "## Maximum numerical errors",
        "",
        f"- Mapping/aggregation: `{_fmt(errors.get('maximum_mapping_or_aggregation_error_m3'))} m³`",
        f"- API volume: `{_fmt(errors.get('maximum_api_volume_error_m3'))} m³`",
        f"- VIC water balance: `{_fmt(errors.get('maximum_vic_water_error_mm'))} mm`",
        f"- FLOW-JA-FACE pair antisymmetry: `{_fmt(errors.get('maximum_lateral_pair_antisymmetry_m3_day'))} m³/day`",
        f"- Connected-cell budget residual: `{_fmt(errors.get('maximum_cell_budget_residual_m3'))} m³`",
        "",
        "## Runtime",
        "",
        f"- Total wall time: `{_fmt(summary.get('total_wall_seconds'))} s`",
    ]
    for key in ("vic", "mf6", "mapping", "mpi", "io"):
        if key in timing:
            lines.append(f"- {key.upper()}: `{_fmt(timing.get(key))} s`")

    lines.extend(["", "## Tables", ""])
    lines.extend(f"- [`tables/{name}`](tables/{name})" for name in table_names)
    lines.extend(["", "## Figures", ""])
    for name in figure_names:
        if name.lower().endswith(".png"):
            lines.append(f"![{name}](figures/{name})")
            lines.append("")

    if notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    lines.extend(
        [
            "## Acceptance details",
            "",
            "```text",
            format_acceptance(acceptance).rstrip(),
            "```",
            "",
        ]
    )
    paths.report_markdown.write_text("\n".join(lines), encoding="utf-8")


def _union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def _exchange_direction(vic: dict[str, Any]) -> str:
    positive = float(vic.get("positive_m3") or 0.0) > 0.0
    negative = float(vic.get("negative_m3") or 0.0) < 0.0
    if positive and negative:
        return "bidirectional"
    if positive:
        return "VIC-to-MF6 only"
    if negative:
        return "MF6-to-VIC only"
    return "none"
