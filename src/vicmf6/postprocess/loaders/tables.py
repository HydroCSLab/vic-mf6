"""Read recorded coupling windows and immutable exchange metadata.

Postprocessing reads completed-run products; it never advances a model."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ...exchange import ExchangeTable

_NUMERIC_WINDOW_FIELDS = {
    "step": int,
    "nonlinear_iterations_max": int,
}


def load_coupling_windows(path: Path) -> list[dict[str, Any]]:
    """read runtime window diagnostics and convert numeric fields eagerly."""

    if not path.is_file():
        raise PostprocessingError(f"coupling diagnostics were not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise PostprocessingError(f"coupling diagnostics contain no rows: {path}")

    converted: list[dict[str, Any]] = []
    required = {"step", "start", "end"} | {
        f"{prefix}_{sign}_m3"
        for prefix in ("vic", "mapped", "boundary", "applied")
        for sign in ("positive", "negative", "net")
    }
    for row_index, row in enumerate(rows, start=2):
        missing = required - row.keys()
        if missing:
            raise PostprocessingError(
                f"missing diagnostic columns in {path}: {sorted(missing)}"
            )
        item: dict[str, Any] = {}
        for key, raw in row.items():
            if raw is None:
                item[key] = None
                continue
            text = raw.strip()
            if key in {"start", "end"}:
                item[key] = text
            elif text == "":
                item[key] = None
            elif key in _NUMERIC_WINDOW_FIELDS:
                try:
                    item[key] = _NUMERIC_WINDOW_FIELDS[key](text)
                except ValueError as exc:
                    raise PostprocessingError(
                        f"invalid {key} in {path} row {row_index}: {text!r}"
                    ) from exc
            else:
                try:
                    item[key] = float(text)
                except ValueError:
                    item[key] = text
        for field in required - {"start", "end"}:
            if not isinstance(item[field], int | float) or not math.isfinite(
                item[field]
            ):
                raise PostprocessingError(
                    f"missing or non-finite {field} in {path} row {row_index}"
                )
        converted.append(item)
    return converted


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PostprocessingError(f"JSON output was not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PostprocessingError(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PostprocessingError(f"expected a JSON object in {path}")
    return value


def load_exchange_records(path: Path) -> list[dict[str, Any]]:
    """read the static overlap table for reporting and plotting metadata."""

    if not path.is_file():
        raise PostprocessingError(f"exchange table was not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise PostprocessingError(f"exchange table contains no overlap rows: {path}")

    integer_fields = {"vic_row", "vic_col", "mf6_node"}
    float_fields = {
        "vic_area_m2",
        "vic_lat",
        "vic_lon",
        "vic_interface_elevation_m",
        "mf6_area_m2",
        "overlap_area_m2",
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, raw in row.items():
            text = "" if raw is None else raw.strip()
            if key in integer_fields and text:
                item[key] = int(text)
            elif key in float_fields:
                item[key] = None if not text else float(text)
            else:
                item[key] = text
        result.append(item)
    return result


def exchange_table_object(config: ApplicationConfig) -> ExchangeTable:
    return ExchangeTable.from_csv(
        config.coupling.exchange_table,
        require_full_vic_coverage=config.coupling.require_full_vic_coverage,
        coverage_relative_tolerance=config.coupling.coverage_relative_tolerance,
    )
