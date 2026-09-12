#!/usr/bin/env python3

"""check high-value conservation signatures from the Stehekin acceptance run."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

MAPPING_TOLERANCE_M3 = 1.0e-5
API_VOLUME_TOLERANCE_M3 = 1.0e-5
LATERAL_TOLERANCE_M3 = 1.0e-5
VIC_WATER_TOLERANCE_MM = 1.0e-8


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_stehekin_result.py DIAGNOSTICS_DIR", file=sys.stderr)
        return 2
    diagnostics = Path(sys.argv[1]).resolve()
    window_path = diagnostics / "coupling_windows.csv"
    summary_path = diagnostics / "run_summary.json"
    if not window_path.is_file() or not summary_path.is_file():
        print(f"acceptance outputs are missing under {diagnostics}", file=sys.stderr)
        return 2

    with window_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if len(rows) != 10:
        failures.append(f"expected 10 daily coupling windows, found {len(rows)}")

    for row in rows:
        step = row["step"]
        for field in (
            "positive_mapping_error_m3",
            "negative_mapping_error_m3",
            "net_mapping_error_m3",
        ):
            if abs(float(row[field])) > MAPPING_TOLERANCE_M3:
                failures.append(f"step {step} {field}={row[field]}")
        if abs(float(row["aggregation_net_error_m3"])) > MAPPING_TOLERANCE_M3:
            failures.append(
                f"step {step} aggregation_net_error_m3={row['aggregation_net_error_m3']}"
            )
        for field in ("positive_api_error_m3", "negative_api_error_m3", "net_api_error_m3"):
            if abs(float(row[field])) > API_VOLUME_TOLERANCE_M3:
                failures.append(f"step {step} {field}={row[field]}")
        water_error = row["maximum_vic_water_error_mm"]
        if water_error and abs(float(water_error)) > VIC_WATER_TOLERANCE_MM:
            failures.append(f"step {step} VIC water error={water_error} mm")
        lateral_net = row["lateral_domain_net_m3"]
        if lateral_net and abs(float(lateral_net)) > LATERAL_TOLERANCE_M3:
            failures.append(f"step {step} lateral domain net={lateral_net} m3")
        lateral_pair_error = row["lateral_maximum_pair_antisymmetry_m3"]
        if (
            lateral_pair_error
            and abs(float(lateral_pair_error)) > LATERAL_TOLERANCE_M3
        ):
            failures.append(
                f"step {step} lateral pair error={lateral_pair_error} m3"
            )

    cumulative = summary["cumulative"]
    mapped_net = float(cumulative["mapped"]["net_m3"])
    boundary_net = float(cumulative["boundary"]["net_m3"])
    applied_net = float(cumulative["applied"]["net_m3"])
    if abs(mapped_net - boundary_net) > MAPPING_TOLERANCE_M3:
        failures.append(
            f"cumulative mapped/boundary net mismatch={boundary_net - mapped_net} m3"
        )
    if abs(boundary_net - applied_net) > API_VOLUME_TOLERANCE_M3:
        failures.append(
            f"cumulative boundary/applied interface mismatch={applied_net - boundary_net} m3"
        )

    positive = float(cumulative["vic"]["positive_m3"])
    negative = float(cumulative["vic"]["negative_m3"])
    if positive <= 0.0 or negative >= 0.0:
        failures.append(
            "the acceptance run did not exercise both signed exchange directions"
        )

    if failures:
        print("[FAIL] Stehekin coupled acceptance")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[OK] Stehekin coupled acceptance")
    print(f"  windows               = {len(rows)}")
    print(f"  cumulative VIC net    = {float(cumulative['vic']['net_m3']):.12e} m3")
    print(f"  cumulative MF6 applied= {applied_net:.12e} m3")
    print(f"  interface error       = {applied_net - mapped_net:.6e} m3")
    print(f"  gross positive        = {positive:.12e} m3")
    print(f"  gross negative        = {negative:.12e} m3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
