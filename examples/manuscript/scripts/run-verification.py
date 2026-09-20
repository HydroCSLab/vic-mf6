#!/usr/bin/env python3
"""Rerun the original numerical experiments with the pinned installed models.

All model workspaces are created below a new output directory. The verification
scripts retain their original calculations, with portable paths and explicit
MPI initialization for the parallel MODFLOW library shipped in the bundle.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import netCDF4 as nc
import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, default=Path("/opt/vicmf6"))
    parser.add_argument("--through", choices=["H2", "H3", "H8"], default="H8",
                        help="Stop after an early stage for installation diagnostics.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    installed = args.install_dir.resolve()
    source = Path(__file__).resolve().parents[1] / "verification"
    sample = out / "sample"
    params = sample / "parameters"
    params.mkdir(parents=True)
    vic = out / "vic-source"
    shutil.copytree(installed / "src/vic", vic,
                    ignore=shutil.ignore_patterns(".git", "*.o", "__pycache__"))
    binary = vic / "vic/drivers/image/vic_image.exe"
    if binary.exists():
        binary.unlink()
    binary.symlink_to(installed / "bin/vic_image.exe")
    fixture_dir = vic / "experiments/gw_exchange"
    fixture_builder = source / "create_mf6_fixtures.py"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(fixture_builder), "--output-dir", str(fixture_dir)],
        check=True,
    )
    bundled = installed / "examples/stehekin"
    inputs = bundled / "input"
    for old, new in [("domain_stehekin_20151028.nc", "domain.stehekin.20151028.nc"),
                     ("stehekin_parameters_20160327.nc", "Stehekin_test_params_20160327.nc")]:
        shutil.copy2(inputs / old, params / new)
    shutil.copy2(inputs / "stehekin_forcings_10_days_1949.nc", params / "forcing_1949.nc")
    template = (bundled / "stehekin.global.txt").read_text()
    template = template.replace("input/domain_stehekin_20151028.nc", str(params / "domain.stehekin.20151028.nc"))
    template = template.replace("input/stehekin_parameters_20160327.nc", str(params / "Stehekin_test_params_20160327.nc"))
    template = template.replace("input/stehekin_forcings_10_days_", str(params / "forcing_"))
    (params / "Stehekin_image_test.global.txt").write_text(template)
    revisions = installed / "share/component-revisions.txt"
    shutil.copy2(revisions, out / "component-revisions.txt")
    hashes = []
    for root in [source, installed / "src/vic/vic", installed / "src/vic-mf6/src", inputs]:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                hashes.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in [Path(__file__), installed / "bin/vic_image.exe", installed / "lib/libmf6.so"]:
        hashes.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in sorted(fixture_dir.rglob("*")):
        if path.is_file():
            hashes.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_csv(out / "source-hashes.csv", hashes)
    env = {k: v for k, v in os.environ.items() if not k.startswith("VIC_GW_")}
    env.update(VICMF6_VERIFICATION_OUTPUT=str(out), VICMF6_VERIFICATION_SAMPLE=str(sample),
               VICMF6_VERIFICATION_VIC=str(vic), LIBMF6=str(installed / "lib/libmf6.so"),
               VICMF6_COMPONENT_REVISIONS=str(revisions), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    logs = out / "logs"
    logs.mkdir()
    executions = []

    def run(name, command, extra=None, cwd=None):
        print(f"Running {name}", flush=True)
        start = time.monotonic()
        with (logs / f"{name}.log").open("w") as log:
            result = subprocess.run(command, env=env | (extra or {}), cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        executions.append(dict(test=name, seconds=time.monotonic() - start, exit_code=result.returncode))
        write_csv(out / "execution.csv", executions)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}")

    def experiment(name, extra=None, label=None):
        run(label or name, [sys.executable, str(source / f"{name}.py")], extra)

    common = dict(VIC_GW_TEST_KA_SCALE="0.001", VIC_GW_MAX_DRAIN_FRACTION="1.0")

    def fixed_vic(name, settings, runoff_steps=24):
        folder = out / name
        folder.mkdir()
        global_file = folder / "global.txt"
        global_file.write_text(template.replace("RESULT_DIR /tmp/vic-output", f"RESULT_DIR {folder}")
                              .replace("RUNOFF_STEPS_PER_DAY  24", f"RUNOFF_STEPS_PER_DAY  {runoff_steps}"))
        write_csv(folder / "environment.csv", [dict(variable=k, value=v) for k, v in settings.items()])
        run(name, ["mpirun", "--oversubscribe", "-n", "1", str(binary), "-g", str(global_file)], settings, params)
        return folder / "fluxes.1949-01-01.nc"

    # Keep ARNO replaced in this control. A negligible positive conductivity
    # satisfies VIC's driver preflight while producing zero exchange in the
    # saved float32 output; turning coupling off would restore ARNO baseflow.
    baseline = fixed_vic("zero-exchange", dict(VIC_GW_FORMULATION="scheidegger",
                         VIC_GW_TEST_GAP_M="105.328", VIC_GW_TEST_KA_SCALE="1e-300"))
    with nc.Dataset(baseline) as ds:
        assert np.ma.abs(ds["OUT_GW_EXCHANGE"][:]).max() == 0
        assert np.ma.abs(ds["OUT_BASEFLOW"][:]).max() == 0
    experiment("h1_read_mf6_head")
    head = float((vic / "experiments/gw_exchange/mf6_h1/mf6_head_offset.txt").read_text())
    head_run = fixed_vic("H1_mf6_to_vic_head", common | dict(VIC_GW_FORMULATION="head",
                         VIC_GW_REFERENCE_DEPTH="base", VIC_GW_TEST_GAP_M="105.328",
                         VIC_GW_HEAD_OFFSET_M=str(head), VIC_GW_EXCHANGE_LENGTH_M="100.0"))
    shutil.copy2(head_run, head_run.parent / "fluxes.mf6_head.nc")
    experiment("h2a_api_flux_test")
    experiment("h2b_vic_flux_to_mf6")
    if args.through == "H2":
        return

    response = []
    with nc.Dataset(baseline) as ds:
        zero_bottom = np.ma.asarray(ds["OUT_SOIL_MOIST"][:, -1])
    for gap in [50.0, 85.012, 105.328, 250.0, 536.425, 750.0]:
        path = fixed_vic(f"lower-boundary-{gap:g}", common | dict(VIC_GW_FORMULATION="scheidegger",
                         VIC_GW_TEST_GAP_M=str(gap), VIC_GW_MAX_DRAIN_FRACTION="0.01"))
        with nc.Dataset(path) as ds:
            q = np.ma.asarray(ds["OUT_GW_EXCHANGE"][:])
            bottom = np.ma.asarray(ds["OUT_SOIL_MOIST"][:, -1])
            values = q.compressed()
            response.append(dict(groundwater_gap_m=gap,
                cumulative_exchange_mm=float(np.ma.mean(q, axis=(-2, -1)).sum()),
                upward_cell_fraction=float(np.mean(values < -1e-10)),
                downward_cell_fraction=float(np.mean(values > 1e-10)),
                initial_bottom_soil_water_mm=float(bottom[0].mean()),
                final_bottom_soil_water_mm=float(bottom[-1].mean()),
                bottom_soil_water_change_mm=float((bottom[-1] - zero_bottom[-1]).mean()),
                maximum_water_balance_error_mm=float(np.ma.abs(ds["OUT_WATER_ERROR"][:]).max())))
    write_csv(out / "data-lower-boundary-response.csv", response)
    # A--G use prescribed groundwater boundaries before evolving MF6 head.
    # The loops preserve the parameter values in the original experiment decks.
    experiment("initial_response_surface")
    prescribed = []

    def boundary_case(name, settings, steps=24):
        path = fixed_vic(name, settings, steps)
        with nc.Dataset(path) as ds:
            q = np.ma.asarray(ds["OUT_GW_EXCHANGE"][:])
            soil = np.ma.asarray(ds["OUT_SOIL_MOIST"][:, -1])
            error = float(np.ma.abs(ds["OUT_WATER_ERROR"][:]).max())
            assert error < 1e-7, (name, error)
            assert float(soil.min()) >= 0, name
            prescribed.append(dict(case=name, runoff_steps_per_day=steps,
                cumulative_exchange_mm=float(q.mean(axis=(-2, -1)).sum()),
                bottom_soil_water_change_mm=float((soil[-1] - zero_bottom[-1]).mean()),
                maximum_water_balance_error_mm=error))
        return path

    for gap in [105.328, 250.0]:
        settings = dict(VIC_GW_FORMULATION="scheidegger", VIC_GW_TEST_GAP_M=str(gap))
        for scale in [0.001, 0.003, 0.01, 0.03, 0.1]:
            boundary_case(f"C-gap-{gap:g}-conductivity-{scale:g}", settings |
                          dict(VIC_GW_TEST_KA_SCALE=str(scale), VIC_GW_MAX_DRAIN_FRACTION="0.01"))
        for limit in [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]:
            boundary_case(f"D-gap-{gap:g}-limit-{limit:g}", settings |
                          dict(VIC_GW_TEST_KA_SCALE="0.1", VIC_GW_MAX_DRAIN_FRACTION=str(limit)))
        for steps in [24, 48, 96, 192, 384]:
            boundary_case(f"E-gap-{gap:g}-steps-{steps}", settings |
                          dict(VIC_GW_TEST_KA_SCALE="0.1", VIC_GW_MAX_DRAIN_FRACTION="1.0"), steps)
    for gap in [1., 2., 5., 10., 50., 105.328]:
        for reference in ["base", "node"]:
            boundary_case(f"F-gap-{gap:g}-{reference}", common | dict(VIC_GW_FORMULATION="scheidegger",
                          VIC_GW_TEST_GAP_M=str(gap), VIC_GW_REFERENCE_DEPTH=reference))
    for pressure in [-536.425, -250., -105.328, -50., 0., 1., 10.]:
        path = boundary_case(f"G2-head-{pressure:g}", common | dict(VIC_GW_FORMULATION="head",
                             VIC_GW_REFERENCE_DEPTH="base", VIC_GW_TEST_GAP_M="105.328",
                             VIC_GW_HEAD_OFFSET_M=str(pressure), VIC_GW_EXCHANGE_LENGTH_M="100.0"))
        if pressure == head:
            with nc.Dataset(path) as prescribed_ds, nc.Dataset(head_run) as live_ds:
                for name in ["OUT_GW_EXCHANGE", "OUT_SOIL_MOIST", "OUT_RUNOFF", "OUT_EVAP"]:
                    np.testing.assert_array_equal(prescribed_ds[name][:], live_ds[name][:])
    write_csv(out / "data-prescribed-boundary-checks.csv", prescribed)
    for name in ["h3a_restart_equivalence", "h3b_closed_loop", "h3c_feedback_analysis"]:
        experiment(name)
    if args.through == "H3":
        return
    experiment("h4_coupling_interval")
    experiment("h5b_12h_restart_equivalence")
    for storage in ["1e-3", "1e-4", "1e-5", "1e-6"]:
        experiment("h5c_subdaily_coupling", dict(H5C_SS_PER_M=storage,
                   H5C_INTERVAL_HOURS="24,12,6,3,1" if storage == "1e-3" else "24,1"),
                   label=f"h5c-{storage}")
    for storage in ["1e-5", "1e-6"]:
        experiment("h6c_midpoint_10day", dict(H6C_SS_PER_M=storage), label=f"h6c-{storage}")
    for name in ["h7a_synthetic_mapper", "h7b_stehekin_real_mapper", "h7c_spatial_api_12cell",
                 "h7d0_spatial_head_equivalence", "h7d_spatial_closed_loop", "h7e_spatial_10day_closed_loop",
                 "h8a_connected_mf6_lateral", "h8b_connected_interface_budget", "h8c_connected_10day_closed_loop"]:
        experiment(name)
    (out / "completion.txt").write_text("All numerical experiment commands completed.\n")


if __name__ == "__main__":
    main()
