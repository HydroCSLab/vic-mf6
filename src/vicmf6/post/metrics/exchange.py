"""Reconstruct signed transfer from VIC fields and native API budget records.

Every comparison uses integrated volume over the same coupling interval.
Rates and depths are converted before aggregation, never averaged together."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ...exchange import ExchangeTable, SignedVolume
from ..records import Mf6Geometry
from .time_series import _direction, _duration_days, _elapsed_days


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
