"""Numerical-contract acceptance checks for one completed coupled run."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from ..config import ApplicationConfig
from ..exchange.conservation import volume_tolerance_m3 as _combined_tolerance
from ..schedule import build_windows
from .records import AcceptanceCheck, AcceptanceResult

_VIC_WATER_TOLERANCE_MM = 1.0e-8
_LATERAL_TOLERANCE_M3 = 1.0e-5
_CELL_BUDGET_TOLERANCE_M3 = 1.0e-5
# Mapping and API checks compare the same values through deterministic array
# operations.  A whole-model CBC balance instead accumulates iterative solver
# budgets over many native steps.  Keep its numerical floor separate so a
# scientifically negligible cumulative roundoff residual is not mislabeled as
# a transfer-conservation failure.
_MF6_DOMAIN_BUDGET_ABSOLUTE_TOLERANCE_M3 = 1.0e-2
_MF6_DOMAIN_BUDGET_RELATIVE_TOLERANCE = 1.0e-8


def evaluate_acceptance(
    config: ApplicationConfig,
    window_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> AcceptanceResult:
    """Evaluate only current-run numerical invariants.

    Historical trajectory reproduction is intentionally not an acceptance
    criterion.  A run is accepted when the executed coupling conserves water,
    the requested API flux is the flux MF6 actually applies, and the MF6 budget
    closes using the current run's own outputs.
    """

    checks = [
        *check_window_conservation(config, window_rows),
        *check_cumulative_exchange(config, summary),
        *check_domain_mass_balance(config, summary),
        *check_lateral_and_cell_budgets(summary),
    ]
    return AcceptanceResult(
        passed=all(check.passed for check in checks), checks=tuple(checks)
    )


def check_window_conservation(
    config: ApplicationConfig, window_rows: list[dict[str, Any]]
) -> list[AcceptanceCheck]:
    windows = build_windows(
        config.coupling.start_time,
        config.coupling.end_time,
        config.coupling.interval_days,
    )
    checks = [
        _equal_check(
            "coupling window count",
            len(window_rows),
            len(windows),
            "diagnostics must contain every configured window",
        )
    ]
    sequence_matches = len(window_rows) == len(windows) and all(
        row.get("step") == window.index + 1
        and row.get("start") == window.start.isoformat()
        and row.get("end") == window.end.isoformat()
        for row, window in zip(window_rows, windows, strict=False)
    )
    checks.append(
        _equal_check(
            "coupling window sequence",
            sequence_matches,
            True,
            "window identifiers and timestamps must match the configured schedule",
        )
    )

    groups = (
        (
            "mapping and aggregation conservation",
            (
                ("positive_mapping_error_m3", "mapped_positive_m3", "vic_positive_m3"),
                ("negative_mapping_error_m3", "mapped_negative_m3", "vic_negative_m3"),
                ("net_mapping_error_m3", "mapped_net_m3", "vic_net_m3"),
                ("aggregation_net_error_m3", "boundary_net_m3", "mapped_net_m3"),
            ),
        ),
        (
            "API volume application",
            tuple(
                (f"{sign}_api_error_m3", f"applied_{sign}_m3", f"boundary_{sign}_m3")
                for sign in ("positive", "negative", "net")
            ),
        ),
    )
    for name, comparisons in groups:
        try:
            evidence = []
            for row in window_rows:
                for error, actual, expected in comparisons:
                    reported_error = float(row[error])
                    actual_volume = float(row[actual])
                    expected_volume = float(row[expected])
                    if not all(
                        map(
                            math.isfinite,
                            (reported_error, actual_volume, expected_volume),
                        )
                    ):
                        raise ValueError("non-finite evidence")
                    allowed = _combined_tolerance(
                        actual_volume,
                        expected_volume,
                        config.coupling.conservation_absolute_tolerance_m3,
                        config.coupling.conservation_relative_tolerance,
                    )
                    discrepancy = max(
                        abs(reported_error), abs(actual_volume - expected_volume)
                    )
                    evidence.append((discrepancy, allowed))
            if not evidence or not all(
                math.isfinite(value) for pair in evidence for value in pair
            ):
                raise ValueError("missing or non-finite evidence")
        except (KeyError, TypeError, ValueError):
            checks.append(
                _missing_check(
                    name,
                    "finite errors and source/target volumes are required for every window",
                )
            )
            continue
        # Report the comparison closest to violating its own tolerance. A large
        # transfer must not lend its relative tolerance to a smaller transfer.
        error, allowed = max(evidence, key=lambda pair: pair[0] - pair[1])
        checks.append(
            _ceiling_check(
                name,
                error,
                allowed,
                "worst window comparison using the runtime absolute + relative tolerance",
            )
        )

    for name, field, tolerance in (
        (
            "API rate application",
            "maximum_api_rate_error_m3_per_day",
            config.coupling.api_absolute_tolerance_m3_per_day,
        ),
        ("VIC water balance", "maximum_vic_water_error_mm", _VIC_WATER_TOLERANCE_MM),
    ):
        try:
            values = [abs(float(row[field])) for row in window_rows]
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError("missing or non-finite evidence")
        except (KeyError, TypeError, ValueError):
            checks.append(
                _missing_check(name, f"finite {field} is required for every window")
            )
            continue
        checks.append(_ceiling_check(name, max(values), tolerance, f"maximum {field}"))
    return checks


def check_cumulative_exchange(
    config: ApplicationConfig, summary: dict[str, Any]
) -> list[AcceptanceCheck]:
    checks: list[AcceptanceCheck] = []
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

    return checks


def check_domain_mass_balance(
    config: ApplicationConfig, summary: dict[str, Any]
) -> list[AcceptanceCheck]:
    checks: list[AcceptanceCheck] = []
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
                    max(
                        config.coupling.conservation_absolute_tolerance_m3,
                        _MF6_DOMAIN_BUDGET_ABSOLUTE_TOLERANCE_M3,
                    ),
                    max(
                        config.coupling.conservation_relative_tolerance,
                        _MF6_DOMAIN_BUDGET_RELATIVE_TOLERANCE,
                    ),
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
                    max(
                        config.coupling.conservation_absolute_tolerance_m3,
                        _MF6_DOMAIN_BUDGET_ABSOLUTE_TOLERANCE_M3,
                    ),
                    max(
                        config.coupling.conservation_relative_tolerance,
                        _MF6_DOMAIN_BUDGET_RELATIVE_TOLERANCE,
                    ),
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

    return checks


def check_lateral_and_cell_budgets(summary: dict[str, Any]) -> list[AcceptanceCheck]:
    checks: list[AcceptanceCheck] = []
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

    return checks


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
        passed=math.isfinite(actual)
        and math.isfinite(ceiling)
        and abs(actual) <= ceiling,
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
        passed=all(map(math.isfinite, (actual, expected, allowed)))
        and abs(actual - expected) <= allowed,
        actual=actual,
        expected=expected,
        tolerance=allowed,
        detail=detail,
    )


def _missing_check(name: str, detail: str) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=name,
        passed=False,
        actual="not available",
        expected="required numerical evidence",
        tolerance=None,
        detail=detail,
    )
