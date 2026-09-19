#!/usr/bin/env python3
"""Collect fresh numerical and process evidence as compact, full-precision CSV."""
import argparse
from pathlib import Path
import shutil

import netCDF4 as nc
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.run_dir.resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = []

    def save(data, name, source):
        data.to_csv(out / name, index=False)
        sources.append(dict(table=name, source=str(Path(source).relative_to(root))))

    def copy(source, name=None):
        save(pd.read_csv(source), name or source.name, source)

    def single(pattern):
        matches = sorted(root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"Expected one fresh result for {pattern}; found {len(matches)}")
        return matches[0]

    numerical = root / "verification"
    if numerical.exists():
        if not (numerical / "completion.txt").exists():
            raise ValueError("Numerical verification did not complete")
        copy(numerical / "data-lower-boundary-response.csv")
        copy(single("verification/H5c_subdaily_coupling_SS_1e-03_*/summary.csv"), "data-h5-subdaily-coupling.csv")
        storage, midpoint = [], []
        for ss in [1e-3, 1e-4, 1e-5, 1e-6]:
            path = single(f"verification/H5c_subdaily_coupling_SS_{ss:.0e}_*/summary.csv")
            rows = pd.read_csv(path)
            daily = rows.loc[rows.interval_hours == 24].iloc[0]
            storage.append(dict(specific_storage_per_m=ss, abs_daily_vs_hourly_error_percent=abs(daily.volume_diff_vs_1h_percent)))
        save(pd.DataFrame(storage), "data-h6-storage-stiffness.csv", numerical)
        for ss in [1e-5, 1e-6]:
            path = single(f"verification/H6c_midpoint_10day_SS_{ss:.0e}_*/summary.csv")
            row = pd.read_csv(path).iloc[0]
            midpoint.append(dict(specific_storage_per_m=ss, explicit_error_percent=abs(row.explicit_volume_error_percent),
                midpoint_error_percent=abs(row.midpoint_volume_error_percent), improvement_factor=row.midpoint_improvement_factor))
        save(pd.DataFrame(midpoint), "data-h6-picard-improvement.csv", numerical)
        for pattern, name in [("H8a_connected_mf6_lateral_*/daily.csv", "data-h8a-daily.csv"),
                              ("H8a_connected_mf6_lateral_*/summary.csv", "data-h8a-summary.csv"),
                              ("H8c_connected_10day_closed_loop_*/daily.csv", "data-h8c-daily.csv")]:
            path = single("verification/" + pattern)
            frame = pd.read_csv(path).drop(columns=["vic_state_file"], errors="ignore")
            save(frame, name, path)
        path = numerical / "H1_mf6_to_vic_head/fluxes.mf6_head.nc"
        with nc.Dataset(path) as ds:
            variables = {"domain_mean_precipitation_mm_day": "OUT_PREC", "domain_mean_air_temperature_c": "OUT_AIR_TEMP",
                         "domain_mean_shortwave_w_m2": "OUT_SWDOWN", "domain_mean_wind_m_s": "OUT_WIND"}
            rows = {"date": pd.date_range("1949-01-01", periods=10).strftime("%Y-%m-%d")}
            rows.update({key: np.ma.mean(ds[value][:], axis=(-2, -1)).filled(np.nan) for key, value in variables.items()})
        save(pd.DataFrame(rows), "data-forcing-10day-summary.csv", path)
    acceptance = root / "acceptance/postprocessing/tables"
    if acceptance.exists():
        for name in ["vic_cells", "mf6_cells", "exchange_overlaps", "mf6_lateral_pairs", "mf6_head_summary", "vic_exchange_daily"]:
            path = acceptance / f"{name}.csv"
            frame = pd.read_csv(path).drop(columns=["source_files"], errors="ignore")
            save(frame, "data-" + name.replace("_", "-") + ".csv", path)
        for path in sorted((root / "groundwater-control").glob("data-*.csv")):
            copy(path)
    for directory, prefix, files in [
        ("reference", "coupled-reference", ["summary", "traces"]),
        ("nonlinear", "nonlinear-interface", ["summary", "traces", "parameters"]),
    ]:
        if not (root / directory).exists():
            continue
        for suffix in files:
            source_name = ("reference-" if directory == "reference" else "nonlinear-") + suffix + ".csv"
            if suffix == "parameters":
                source_name = "parameters.csv"
            copy(root / directory / source_name, f"data-{prefix}-{suffix}.csv")
    if (root / "nonlinear/kernel-checks.csv").exists():
        copy(root / "nonlinear/kernel-checks.csv", "data-nonlinear-kernel-checks.csv")
    # The campaign analyzers write directly to run-dir/tables.
    if (root / "tables").resolve() != out:
        for path in sorted((root / "tables").glob("data-*.csv")):
            copy(path)
    pd.DataFrame(sources).to_csv(out / "table-provenance.csv", index=False)
    print(f"Collected {len(sources)} tables in {out}")


if __name__ == "__main__":
    main()
