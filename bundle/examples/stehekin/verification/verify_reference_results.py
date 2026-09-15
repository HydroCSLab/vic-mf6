#!/usr/bin/env python3
"""Audit compact numerical evidence from the staged VIC--MF6 campaign.

This program deliberately checks recorded reference results rather than
pretending that a table is a fresh model run.  ``scripts/run-acceptance.sh``
is the executable test of the shipped VIC, MODFLOW 6, and MPI coupler.  This
audit makes the broader H1--H8 evidence reviewable without distributing large
binary budget files, NetCDF restart chains, and local-path test scripts.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference-results.json"


def check(condition: bool, description: str, checks: list[tuple[str, bool]]) -> None:
    """Append one named contract so all failures are reported together."""
    checks.append((description, bool(condition)))


def strictly_decreases(values: list[float]) -> bool:
    """Return true when every finite-resolution value improves monotonically."""
    return all(right < left for left, right in zip(values, values[1:]))


def strictly_increases(values: list[float]) -> bool:
    """Return true when every successive value increases."""
    return all(right > left for left, right in zip(values, values[1:]))


def main() -> None:
    evidence = json.loads(REFERENCE.read_text(encoding="utf-8"))
    checks: list[tuple[str, bool]] = []

    api = evidence["api_sign"]
    check(len(api) == 2, "H2a records both API signs", checks)
    check(all(abs(row["target_m3_day"] - row["actual_m3_day"]) <= 1e-12 for row in api),
          "H2a API target equals applied flow", checks)
    check(all(abs(row["head_error_m"]) <= 1e-12 for row in api),
          "H2a API head response", checks)

    h2b = evidence["vic_to_mf6"]
    check(h2b["active_vic_cells"] == 16 and h2b["active_area_m2"] > 2e9,
          "H2b Stehekin source geometry", checks)
    check(abs(h2b["relative_volume_error"]) < 1e-10,
          "H2b VIC volume equals MODFLOW storage", checks)

    h3 = evidence["closed_loop"]
    check(h3["days"] == 10 and abs(h3["relative_volume_error"]) < 1e-12,
          "H3 persistent closed-loop conservation", checks)
    check(h3["maximum_vic_water_error_mm"] < 1e-10,
          "H3 VIC water balance", checks)

    response = evidence["lower_boundary_response"]
    exchange = [row["transfer_mm"] for row in response]
    check(strictly_decreases([-value for value in exchange[:3]]) and strictly_increases(exchange[3:]),
          "VIC lower-boundary response approaches and passes through equilibrium", checks)
    check(exchange[0] < 0 < exchange[-1], "VIC lower boundary supports both directions", checks)
    check(all(abs(row["transfer_mm"] + row["bottom_soil_change_mm"]) < 2e-4 for row in response),
          "VIC lower-boundary transfer updates bottom-soil storage", checks)

    temporal = evidence["temporal_intervals"]
    check(strictly_decreases([row["absolute_error_percent"] for row in temporal[:-1]]),
          "H5 shorter coupling windows reduce splitting error", checks)
    check(max(abs(row["relative_conservation_error"]) for row in temporal) < 1e-10,
          "H5 conservation is independent of interval", checks)

    picard = evidence["midpoint_picard"]
    check(all(row["midpoint_error_percent"] < row["explicit_error_percent"] for row in picard),
          "H6 midpoint Picard reduces time-lag error", checks)
    check(all(row["improvement_factor"] > 100 for row in picard),
          "H6 midpoint Picard improves both stiff cases by more than 100-fold", checks)

    mapping = evidence["spatial_mapping"]
    check((mapping["vic_cells"], mapping["mf6_nodes"], mapping["intersections"]) == (16, 12, 42),
          "H7 nonmatching-grid fixture dimensions", checks)
    check(mapping["coverage_error_m2"] < 1e-6 and abs(mapping["net_mapping_error_m3"]) < 1e-10,
          "H7 overlap coverage and net volume", checks)
    check(mapping["constant_head_error_m"] < 1e-10,
          "H7 reverse head mapping", checks)

    h8 = evidence["connected_groundwater"]
    check(h8["days"] == 10 and h8["nja"] == 46, "H8 connected groundwater fixture", checks)
    check(abs(h8["relative_volume_error"]) < 1e-12,
          "H8 connected cross-model conservation", checks)
    check(h8["maximum_pair_antisymmetry_m3_day"] < 1e-10 and h8["maximum_lateral_domain_volume_m3"] < 1e-8,
          "H8 internal lateral-flow cancellation", checks)
    check(h8["maximum_cell_budget_residual_m3"] < 1e-5,
          "H8 cellwise groundwater budget", checks)

    width = max(len(name) for name, _ in checks)
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL':4s}  {name:<{width}s}")
    if not all(passed for _, passed in checks):
        raise SystemExit("reference-evidence audit failed")
    print(f"[OK] {len(checks)} staged verification contracts passed")


if __name__ == "__main__":
    main()
