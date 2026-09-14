"""Numerical-contract acceptance checks for one completed coupled run."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..config import ApplicationConfig
from .types import AcceptanceCheck, AcceptanceResult

_VIC_WATER_TOLERANCE_MM = 1.0e-8
_LATERAL_TOLERANCE_M3 = 1.0e-5
_CELL_BUDGET_TOLERANCE_M3 = 1.0e-5


def evaluate_acceptance(
    config: ApplicationConfig,
    window_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    exchange_table_vic_cells: int | None = None,
    exchange_table_overlap_rows: int | None = None,
    mf6_cell_count: int | None = None,
    reference_path: object | None = None,
) -> AcceptanceResult:
    """Evaluate only current-run numerical invariants.

    Historical trajectory reproduction is intentionally not an acceptance
    criterion.  A run is accepted when the executed coupling conserves water,
    the requested API flux is the flux MF6 actually applies, and the MF6 budget
    closes using the current run's own outputs.
    """

    # Backward compatibility with the first comprehensive postprocessor.
    # These values were used only by the archived H8c regression checks.
    # Keep accepting them so existing callers/tests remain valid, but do not
    # use them in current-run numerical acceptance.
    _ = (
        exchange_table_vic_cells,
        exchange_table_overlap_rows,
        mf6_cell_count,
        reference_path,
    )

    checks: list[AcceptanceCheck] = []

    expected_windows = int(
        round(
            (config.coupling.end_time - config.coupling.start_time).total_seconds()
            / 86400.0
            / config.coupling.interval_days
        )
    )
    checks.append(
        _equal_check(
            "coupling window count",
            len(window_rows),
            expected_windows,
            "the completed diagnostics must contain every configured window",
        )
    )

    maximum_mapping = max(
        abs(float(row.get(name) or 0.0))
        for row in window_rows
        for name in (
            "positive_mapping_error_m3",
            "negative_mapping_error_m3",
            "net_mapping_error_m3",
            "aggregation_net_error_m3",
        )
    )
    checks.append(
        _ceiling_check(
            "mapping and aggregation conservation",
            maximum_mapping,
            config.coupling.conservation_absolute_tolerance_m3,
            "maximum absolute runtime mapping/aggregation error",
        )
    )

    maximum_api_volume = max(
        abs(float(row.get(name) or 0.0))
        for row in window_rows
        for name in (
            "positive_api_error_m3",
            "negative_api_error_m3",
            "net_api_error_m3",
        )
    )
    checks.append(
        _ceiling_check(
            "API volume application",
            maximum_api_volume,
            config.coupling.conservation_absolute_tolerance_m3,
            "node-target versus converged API SIMVALS volume error",
        )
    )

    maximum_api_rate = max(
        abs(float(row.get("maximum_api_rate_error_m3_per_day") or 0.0))
        for row in window_rows
    )
    checks.append(
        _ceiling_check(
            "API rate application",
            maximum_api_rate,
            config.coupling.api_absolute_tolerance_m3_per_day,
            "maximum per-node API rate mismatch after finalize_solve",
        )
    )

    water_values = [
        abs(float(row["maximum_vic_water_error_mm"]))
        for row in window_rows
        if row.get("maximum_vic_water_error_mm") is not None
    ]
    if water_values:
        checks.append(
            _ceiling_check(
                "VIC water balance",
                max(water_values),
                _VIC_WATER_TOLERANCE_MM,
                "maximum absolute OUT_WATER_ERROR over coupling windows",
            )
        )
    else:
        checks.append(
            _missing_check(
                "VIC water balance",
                "OUT_WATER_ERROR must be available to establish VIC-side closure",
            )
        )

    cumulative_vic = summary["exchange_m3"]["vic"]
    cumulative_target = summary["exchange_m3"]["node_target"]
    checks.append(
        _close_check(
            "cumulative VIC to node-target net volume",
            float(cumulative_target["net_m3"]),
            float(cumulative_vic["net_m3"]),
            config.coupling.conservation_absolute_tolerance_m3,
            config.coupling.conservation_relative_tolerance,
            "net transfer must survive overlap-to-node aggregation",
        )
    )

    api_applied = summary["exchange_m3"].get("api_applied")
    if api_applied is None:
        checks.append(
            _missing_check(
                "cumulative node-target to API-applied net volume",
                "saved MF6 API/SIMVALS evidence is required to prove the requested boundary was applied",
            )
        )
    else:
        checks.append(
            _close_check(
                "cumulative node-target to API-applied net volume",
                float(api_applied["net_m3"]),
                float(cumulative_target["net_m3"]),
                config.coupling.conservation_absolute_tolerance_m3,
                config.coupling.conservation_relative_tolerance,
                "saved MF6 API budget must reproduce the node target",
            )
        )

    mass = summary.get("mass_balance_m3") or {}
    if mass.get("available"):
        checks.append(
            _ceiling_check(
                "end-to-end VIC to MF6 mass closure",
                float(mass["end_to_end_residual_m3"]),
                _combined_tolerance(
                    float(mass["mf6_storage_state_change_m3"]),
                    float(mass["vic_interface_net_m3"])
                    + float(mass["mf6_other_nonstorage_net_m3"]),
                    config.coupling.conservation_absolute_tolerance_m3,
                    config.coupling.conservation_relative_tolerance,
                ),
                "MF6 storage change = VIC interface transfer + all other non-storage MF6 terms",
            )
        )
        checks.append(
            _ceiling_check(
                "MF6 CBC domain budget closure",
                float(mass["mf6_cbc_residual_m3"]),
                _combined_tolerance(
                    float(mass["mf6_api_net_m3"])
                    + float(mass["mf6_other_nonstorage_net_m3"]),
                    -float(mass["mf6_storage_budget_net_m3"]),
                    config.coupling.conservation_absolute_tolerance_m3,
                    config.coupling.conservation_relative_tolerance,
                ),
                "sum of MF6 API, storage, and other non-storage CBC terms over the domain",
            )
        )
    else:
        checks.append(
            _missing_check(
                "end-to-end VIC to MF6 mass closure",
                str(
                    mass.get("reason") or "MF6 CBC mass-balance evidence is unavailable"
                ),
            )
        )

    lateral_error = summary["errors"].get("maximum_lateral_pair_antisymmetry_m3_day")
    if lateral_error is not None:
        checks.append(
            _ceiling_check(
                "FLOW-JA-FACE pair antisymmetry",
                float(lateral_error),
                _LATERAL_TOLERANCE_M3,
                "postprocessed off-diagonal FLOW-JA-FACE pairs",
            )
        )

    cell_budget = summary.get("cell_budget_m3") or {}
    lateral_sum = cell_budget.get("lateral_row_sum")
    if lateral_sum is not None:
        checks.append(
            _ceiling_check(
                "domain lateral-flow cancellation",
                float(lateral_sum),
                _LATERAL_TOLERANCE_M3,
                "internal FLOW-JA-FACE transfers must sum to zero over the complete MF6 domain",
            )
        )

    cell_error = summary["errors"].get("maximum_cell_budget_residual_m3")
    if cell_error is not None:
        checks.append(
            _ceiling_check(
                "connected MF6 cell budget",
                float(cell_error),
                _CELL_BUDGET_TOLERANCE_M3,
                "for confined cells, Delta storage = interface + lateral closure",
            )
        )

    return AcceptanceResult(
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def acceptance_as_dict(result: AcceptanceResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "checks": [asdict(check) for check in result.checks],
    }


def format_acceptance(result: AcceptanceResult) -> str:
    lines = [
        "VIC-MODFLOW 6 coupled-run acceptance",
        "===================================",
        "",
        f"numerical contracts : {'PASS' if result.passed else 'FAIL'}",
        "",
    ]

    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"[{status}] {check.name}")
        lines.append(f"       actual   : {check.actual}")
        if check.expected is not None:
            lines.append(f"       expected : {check.expected}")
        if check.tolerance is not None:
            lines.append(f"       tolerance: {check.tolerance:.12e}")
        lines.append(f"       detail   : {check.detail}")
    lines.append("")
    return "\n".join(lines)


def _equal_check(name: str, actual: Any, expected: Any, detail: str) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=name,
        passed=actual == expected,
        actual=actual,
        expected=expected,
        tolerance=None,
        detail=detail,
    )


def _ceiling_check(
    name: str, actual: float, ceiling: float, detail: str
) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=name,
        passed=abs(actual) <= ceiling,
        actual=actual,
        expected=f"abs(value) <= {ceiling:.12e}",
        tolerance=ceiling,
        detail=detail,
    )


def _close_check(
    name: str,
    actual: float,
    expected: float,
    absolute: float,
    relative: float,
    detail: str,
) -> AcceptanceCheck:
    allowed = _combined_tolerance(actual, expected, absolute, relative)
    return AcceptanceCheck(
        name=name,
        passed=abs(actual - expected) <= allowed,
        actual=actual,
        expected=expected,
        tolerance=allowed,
        detail=detail,
    )


def _combined_tolerance(
    actual: float,
    expected: float,
    absolute: float,
    relative: float,
) -> float:
    return float(absolute + relative * max(abs(actual), abs(expected)))


def _missing_check(name: str, detail: str) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=name,
        passed=False,
        actual="not available",
        expected="required numerical evidence",
        tolerance=None,
        detail=detail,
    )
