#!/usr/bin/env python3
"""Run a controlled snowmelt, dry-down, and groundwater-withdrawal experiment.

The original Stehekin files are read only. Every invocation creates a new run
directory containing the generated forcing, complete groundwater input deck,
VIC restart chain, model logs, numerical budgets, and source fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import time

import flopy
import netCDF4 as nc
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
# The image keeps the parent repository's coupler source under src/vic-mf6.
BUNDLE_DIR = EXAMPLE_DIR.parents[1]
START = datetime(1949, 1, 1)
VARIABLES = (
    "OUT_PREC",
    "OUT_RAINF",
    "OUT_SNOWF",
    "OUT_SNOW_MELT",
    "OUT_SWE",
    "OUT_EVAP",
    "OUT_TRANSP_VEG",
    "OUT_EVAP_BARE",
    "OUT_RUNOFF",
    "OUT_BASEFLOW",
    "OUT_GW_EXCHANGE",
    "OUT_WATER_ERROR",
    "OUT_SOIL_MOIST",
    "OUT_ROOTMOIST",
    "OUT_WDEW",
    "OUT_SNOW_CANOPY",
    "OUT_AIR_TEMP",
)


def fingerprint(path: Path) -> dict:
    """Record actual input bytes; a commit alone cannot identify a local build."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def build_geometry(parameter_dir: Path, run_dir: Path, nrow=6, ncol=8) -> dict:
    """Intersect the real VIC footprint with an independent metric DIS grid."""
    domain = parameter_dir / "domain.stehekin.20151028.nc"
    transform = Transformer.from_crs(4326, 5070, always_xy=True)
    with nc.Dataset(domain) as ds:
        mask = np.asarray(ds["mask"][:]) == 1
        latitude = np.asarray(ds["lat"][:])
        longitude = np.asarray(ds["lon"][:])
        areas = np.asarray(ds["area"][:])
    rows, cols = np.where(mask)
    vic_polygons = []
    for row, col in zip(rows, cols, strict=True):
        lon, lat = longitude[col], latitude[row]
        corners = [
            (lon - 0.0625, lat - 0.0625),
            (lon + 0.0625, lat - 0.0625),
            (lon + 0.0625, lat + 0.0625),
            (lon - 0.0625, lat + 0.0625),
        ]
        vic_polygons.append(Polygon([transform.transform(x, y) for x, y in corners]))
    west = min(p.bounds[0] for p in vic_polygons) - 10
    south = min(p.bounds[1] for p in vic_polygons) - 10
    east = max(p.bounds[2] for p in vic_polygons) + 10
    north = max(p.bounds[3] for p in vic_polygons) + 10
    dx, dy = (east - west) / ncol, (north - south) / nrow
    groundwater = [
        box(west + c * dx, north - (r + 1) * dy, west + (c + 1) * dx, north - r * dy)
        for r in range(nrow)
        for c in range(ncol)
    ]
    overlaps = np.array(
        [[v.intersection(g).area for g in groundwater] for v in vic_polygons]
    )
    geometric_areas = np.array([p.area for p in vic_polygons])
    np.testing.assert_allclose(overlaps.sum(axis=1), geometric_areas, rtol=1e-11)
    # vic's authoritative areas determine volume. normalize intersection
    # fractions, rather than quietly substituting a projected polygon area.
    weights = overlaps / geometric_areas[:, None]
    authoritative = areas[rows, cols]
    transfer_areas = weights * authoritative[:, None]
    pumped = np.array([r * ncol + c for r in (2, 3) for c in (3, 4)])
    metadata = dict(
        nrow=nrow,
        ncol=ncol,
        nlay=3,
        west=west,
        south=south,
        east=east,
        north=north,
        dx=dx,
        dy=dy,
        pumped_nodes=pumped.tolist(),
        crs="EPSG:5070",
        interface_elevation_m=0.0,
        layer_bottoms_m=[-20.0, -25.0, -75.0],
    )
    (run_dir / "geometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    features = []
    for kind, polygons in (("vic", vic_polygons), ("mf6", groundwater)):
        for index, polygon in enumerate(polygons):
            features.append(
                dict(
                    type="Feature",
                    geometry=mapping(polygon),
                    properties=dict(grid=kind, node=index),
                )
            )
    (run_dir / "grids.geojson").write_text(
        json.dumps(
            dict(
                type="FeatureCollection",
                features=features,
                crs=dict(type="name", properties=dict(name="EPSG:5070")),
            )
        )
    )
    np.savez(
        run_dir / "mapping.npz",
        weights=weights,
        areas=authoritative,
        transfer_areas=transfer_areas,
        rows=rows,
        cols=cols,
        lat=latitude[rows],
        lon=longitude[cols],
        geometric_areas=geometric_areas,
    )
    return dict(
        **metadata,
        weights=weights,
        areas=authoritative,
        transfer_areas=transfer_areas,
        rows=rows,
        cols=cols,
        lat=latitude[rows],
        lon=longitude[cols],
    )


def build_forcing(
    sample_dir: Path, run_dir: Path, days: int, snowfall_mm_day: float
) -> Path:
    """Prescribe a reproducible cold-to-warm sequence, not observed weather."""
    source = sample_dir / "forcings/Stehekin_image_test.forcings_10days.1949.nc"
    prefix = run_dir / "forcing."
    hours = np.arange(days * 24)
    day = hours / 24
    diurnal = np.sin(2 * np.pi * (hours % 24 - 9) / 24)
    mean_temperature = np.where(
        day < 10, -6.0, np.minimum(20.0, 2.0 + (day - 10) * 1.8)
    )
    temperature = mean_temperature + np.where(day < 10, 2.0, 6.0) * diurnal
    solar = np.maximum(0.0, np.sin(np.pi * ((hours % 24) - 6) / 12))
    prescribed = {
        "tas": temperature,
        "prcp": np.where(day < 10, snowfall_mm_day / 24, 0.0),
        "dswrf": solar * np.where(day < 10, 200.0, 600.0),
        "dlwrf": np.where(day < 10, 230.0, 330.0),
        "vp": np.where(day < 10, 0.25, 1.0),
        "wind": np.full(len(hours), 2.0),
    }
    with nc.Dataset(source) as src, nc.Dataset(str(prefix) + "1949.nc", "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.description = "synthetic 10-day snowfall followed by warming and dry-down"
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(hours) if name == "time" else len(dim))
        for name, var in src.variables.items():
            attrs = {k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"}
            out = dst.createVariable(
                name,
                var.dtype,
                var.dimensions,
                fill_value=getattr(var, "_FillValue", None),
            )
            out.setncatts(attrs)
            if name == "time":
                out[:] = hours
            elif "time" not in var.dimensions:
                out[:] = var[:]
            elif name in prescribed:
                out[:] = np.broadcast_to(prescribed[name][:, None, None], out.shape)
            else:
                out[:] = np.broadcast_to(np.ma.mean(var[:], axis=0), out.shape)
    pd.DataFrame(dict(hour=hours, day=day, **prescribed)).to_csv(
        run_dir / "forcing.csv", index=False
    )
    return prefix


def build_groundwater(args, geometry: dict, run_dir: Path) -> Path:
    """Add an explicit aquitard and deep withdrawal below a convertible aquifer."""
    workspace = run_dir / "mf6"
    sim = flopy.mf6.MFSimulation(sim_name="feedback", sim_ws=workspace)
    count = round(args.days / args.interval)
    flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=count, perioddata=[(args.interval, 1, 1.0)] * count
    )
    flopy.mf6.ModflowIms(
        sim,
        complexity="COMPLEX",
        outer_dvclose=1e-9,
        inner_dvclose=1e-10,
        rcloserecord=1e-5,
        outer_maximum=200,
        inner_maximum=300,
        linear_acceleration="BICGSTAB",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="FLOW", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=3,
        nrow=geometry["nrow"],
        ncol=geometry["ncol"],
        delr=geometry["dx"],
        delc=geometry["dy"],
        top=0.0,
        botm=[-20.0, -25.0, -75.0],
        xorigin=geometry["west"],
        yorigin=geometry["south"],
        length_units="METERS",
    )
    flopy.mf6.ModflowGwfic(gwf, strt=-2.0)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=[1, 0, 0],
        k=[5.0, args.aquitard_k, 10.0],
        k33=[0.5, args.aquitard_k, 1.0],
        save_flows=True,
        save_specific_discharge=True,
    )
    flopy.mf6.ModflowGwfsto(
        gwf, iconvert=[1, 0, 0], ss=1e-5, sy=0.15, transient={0: True}, save_flows=True
    )
    flopy.mf6.ModflowGwfapi(
        gwf,
        pname="EXCHANGE",
        maxbound=geometry["nrow"] * geometry["ncol"],
        save_flows=True,
    )
    withdrawals = {}
    total_rate = args.pumping_mm_day * 0.001 * geometry["areas"].sum()
    for step in range(count):
        elapsed = step * args.interval
        active = 20.0 <= elapsed < 45.0 and args.case in {"pumped", "replay"}
        rate = -total_rate / len(geometry["pumped_nodes"]) if active else 0.0
        withdrawals[step] = [
            ((2, node // geometry["ncol"], node % geometry["ncol"]), rate)
            for node in geometry["pumped_nodes"]
        ]
    flopy.mf6.ModflowGwfwel(
        gwf, pname="WITHDRAWAL", stress_period_data=withdrawals, save_flows=True
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="flow.hds",
        budget_filerecord="flow.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    return workspace


def vic_window(args, geometry, prefix, run_dir, step, state, heads):
    """Advance the compiled Image Driver and require a fresh restart and output."""
    window = run_dir / "vic" / f"window-{step:04d}"
    window.mkdir(parents=True)
    start = START + timedelta(days=step * args.interval)
    stop = start + timedelta(days=args.interval)
    hours = round(args.interval * 24)
    parameter_dir = args.sample_dir / "parameters"
    base = (parameter_dir / "Stehekin_image_test.global.txt").read_text().splitlines()
    excluded = {
        "STARTYEAR",
        "STARTMONTH",
        "STARTDAY",
        "STARTSEC",
        "ENDYEAR",
        "ENDMONTH",
        "ENDDAY",
        "NRECS",
        "RESULT_DIR",
        "OUTFILE",
        "OUTVAR",
        "AGGFREQ",
        "COMPRESS",
        "OUT_FORMAT",
        "INIT_STATE",
        "STATENAME",
        "STATEYEAR",
        "STATEMONTH",
        "STATEDAY",
        "STATESEC",
        "STATE_FORMAT",
    }
    lines = [
        line for line in base if not line.split() or line.split()[0] not in excluded
    ]
    lines = [
        f"FORCING1 {prefix}" if line.split() and line.split()[0] == "FORCING1" else line
        for line in lines
    ]
    lines.extend(
        [
            f"STARTYEAR {start.year}",
            f"STARTMONTH {start.month}",
            f"STARTDAY {start.day}",
            f"STARTSEC {start.hour*3600}",
            f"NRECS {hours}",
            f"RESULT_DIR {window}",
            "OUTFILE hydro",
            f"AGGFREQ NHOURS {hours}",
            "OUT_FORMAT NETCDF4",
            "COMPRESS 1",
        ]
    )
    lines.extend(f"OUTVAR {name} %.10g OUT_TYPE_DOUBLE 1" for name in VARIABLES)
    # state differences require end-of-window storage; VIC's default aggregation
    # for soil water and snow is END, while water fluxes use SUM.
    lines.extend(
        [
            f"STATENAME {window/'state'}",
            f"STATEYEAR {stop.year}",
            f"STATEMONTH {stop.month}",
            f"STATEDAY {stop.day}",
            f"STATESEC {stop.hour*3600}",
            "STATE_FORMAT NETCDF4_CLASSIC",
        ]
    )
    if state is not None:
        lines.append(f"INIT_STATE {state}")
    global_path = window / "global.txt"
    global_path.write_text("\n".join(lines) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VIC_GW_")}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VIC_ALLOW_PARTIAL_DAY="1")
    if args.case != "stock":
        head_file = window / "heads.txt"
        np.savetxt(
            head_file,
            np.column_stack((geometry["lat"], geometry["lon"], heads)),
            fmt="%.17g",
        )
        env.update(
            VIC_GW_FORMULATION="head",
            VIC_GW_REFERENCE_DEPTH="base",
            VIC_GW_HEAD_FILE=str(head_file),
            VIC_GW_EXCHANGE_LENGTH_M=str(args.length),
            VIC_GW_TEST_KA_SCALE=str(args.scale),
            VIC_GW_MAX_DRAIN_FRACTION="1",
        )
    with (window / "vic.log").open("w") as log:
        completed = subprocess.run(
            [str(args.vic_exe), "-g", str(global_path)],
            cwd=parameter_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"VIC exit {completed.returncode}: {window/'vic.log'}")
    outputs = list(window.glob("hydro*.nc"))
    states = list(window.glob("state*.nc"))
    if len(outputs) != 1 or len(states) != 1:
        raise RuntimeError(f"expected one output and one new restart in {window}")
    fields = {}
    with nc.Dataset(outputs[0]) as ds:
        for name in VARIABLES:
            values = np.ma.asarray(ds[name][0])
            selected = values[..., geometry["rows"], geometry["cols"]]
            if np.ma.getmaskarray(selected).any() or not np.isfinite(selected).all():
                raise RuntimeError(f"masked or nonfinite {name} in {outputs[0]}")
            fields[name] = np.asarray(selected, dtype=float)
    return fields, states[0]


def extract_budgets(run_dir: Path, arrays: dict, interval: float) -> pd.DataFrame:
    """Use solved MF6 package budgets, including unconfined storage and wells."""
    budget = flopy.utils.CellBudgetFile(run_dir / "mf6/flow.cbc", precision="double")
    grid = flopy.mf6.utils.MfGrdFile(run_dir / "mf6/FLOW.dis.grb")
    records = []
    vertical = []
    storage = []
    nplane = arrays["heads"].shape[1] // 3
    for step, kstpkper in enumerate(budget.get_kstpkper()):
        stor = sum(
            np.asarray(budget.get_data(kstpkper=kstpkper, text=name)[0]).ravel()
            for name in ("STO-SS", "STO-SY")
        )
        wells = budget.get_data(kstpkper=kstpkper, text="WEL")[0]
        well_by_node = np.zeros_like(stor)
        np.add.at(well_by_node, wells["node"] - 1, wells["q"])
        api = budget.get_data(kstpkper=kstpkper, text="API")[0]
        api_by_node = np.zeros_like(stor)
        np.add.at(api_by_node, api["node"] - 1, api["q"])
        flow = budget.get_data(kstpkper=kstpkper, text="FLOW-JA-FACE")[0].ravel()
        internal = np.zeros_like(stor)
        vertical_down = np.zeros(nplane)
        for node in range(len(stor)):
            for pos in range(grid.ia[node] + 1, grid.ia[node + 1]):
                neighbor = grid.ja[pos]
                internal[node] += flow[pos]
                if node < nplane and neighbor == node + nplane:
                    vertical_down[node] = -flow[pos] * interval
        residual = (stor + well_by_node + api_by_node + internal) * interval
        records.append(
            dict(
                time_days=(step + 1) * interval,
                storage_change_m3=float(-stor.sum() * interval),
                interface_m3=float(api_by_node.sum() * interval),
                withdrawal_m3=float(well_by_node.sum() * interval),
                domain_residual_m3=float(residual.sum()),
                max_cell_residual_m3=float(np.max(np.abs(residual))),
                internal_cancellation_m3=float(internal.sum() * interval),
            )
        )
        vertical.append(vertical_down)
        storage.append(-stor * interval)
    arrays["vertical_down_m3"] = np.array(vertical)
    arrays["storage_change_by_node_m3"] = np.array(storage)
    result = pd.DataFrame(records)
    result.to_csv(run_dir / "budgets.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--case", choices=["baseline", "pumped", "replay", "stock"], default="baseline"
    )
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--aquitard-k", type=float, default=0.02)
    parser.add_argument("--pumping-mm-day", type=float, default=3.0)
    parser.add_argument("--snowfall-mm-day", type=float, default=8.0)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--replay-from", type=Path)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path.home() / "usr/local/src/VIC_sample_data/image/Stehekin",
    )
    parser.add_argument(
        "--vic-exe",
        type=Path,
        default=BUNDLE_DIR / "bin/vic_image.exe",
    )
    parser.add_argument(
        "--mf6-library",
        type=Path,
        default=BUNDLE_DIR / "lib/libmf6.so",
    )
    parser.add_argument(
        "--coupler-dir", type=Path, default=BUNDLE_DIR / "src/vic-mf6"
    )
    args = parser.parse_args()
    if args.days <= 0 or args.interval not in (1.0, 0.5, 0.25) or args.aquitard_k <= 0:
        parser.error(
            "days and conductivity must be positive; interval must be 1, 0.5, or 0.25 day"
        )
    if args.case == "replay" and args.replay_from is None:
        parser.error("--replay-from is required for replay")
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings["inputs"] = [
        fingerprint(path)
        for path in (
            args.vic_exe,
            args.mf6_library,
            args.sample_dir / "parameters/Stehekin_test_params_20160327.nc",
            args.sample_dir / "parameters/domain.stehekin.20151028.nc",
            Path(__file__),
        )
    ]
    (args.run_dir / "provenance.json").write_text(json.dumps(settings, indent=2) + "\n")
    geometry = build_geometry(args.sample_dir / "parameters", args.run_dir)
    prefix = build_forcing(
        args.sample_dir, args.run_dir, args.days, args.snowfall_mm_day
    )
    runtime = None
    nplane = geometry["nrow"] * geometry["ncol"]
    if args.case != "stock":
        workspace = build_groundwater(args, geometry, args.run_dir)
        sys.path.insert(0, str(args.coupler_dir / "src"))
        from mpi4py import MPI
        from vicmf6.config import Mf6Config
        from vicmf6.mf6 import Mf6Runtime

        config = Mf6Config(
            workspace=workspace,
            library=args.mf6_library,
            simulation_namefile="mfsim.nam",
            start_time=START,
            api_packages={"FLOW": "EXCHANGE"},
            solution_ids={"FLOW": 1},
            total_time_days=float(args.days),
            time_step_boundaries_days=tuple(
                np.arange(1, round(args.days / args.interval) + 1) * args.interval
            ),
            max_solve_iterations=200,
        )
        runtime = Mf6Runtime(
            config,
            model_name="FLOW",
            coupled_nodes=np.arange(1, nplane + 1),
            logger=logging.getLogger("feedback"),
        )
        runtime.initialize(MPI.COMM_SELF.py2f())
    replay = np.load(args.replay_from / "fields.npz") if args.case == "replay" else None
    if replay is not None and len(replay["heads"]) != round(args.days / args.interval):
        raise ValueError("replay windows must match the new run exactly")
    records = []
    fields_history = {name: [] for name in VARIABLES}
    head_history = []
    interface_history = []
    state = None
    try:
        for step in range(round(args.days / args.interval)):
            head = (
                runtime.current_head()
                if runtime is not None
                else np.full(3 * nplane, np.nan)
            )
            returned = geometry["weights"] @ head[:nplane]
            if replay is None:
                fields, state = vic_window(
                    args, geometry, prefix, args.run_dir, step, state, returned
                )
            else:
                fields = {name: replay[name][step].copy() for name in VARIABLES}
            volume = fields["OUT_GW_EXCHANGE"] * 0.001 @ geometry["transfer_areas"]
            api_error = 0.0
            if runtime is not None:
                all_nodes = np.zeros(3 * nplane)
                all_nodes[:nplane] = volume
                result = runtime.advance_to(
                    (step + 1) * args.interval, all_nodes, api_tolerance_m3_per_day=1e-5
                )
                head = result.head_m
                api_error = result.maximum_api_error_m3_per_day
            row = dict(
                time_days=(step + 1) * args.interval,
                api_error_m3_day=api_error,
                mapping_error_m3=float(
                    volume.sum()
                    - np.dot(fields["OUT_GW_EXCHANGE"] * 0.001, geometry["areas"])
                ),
                max_vic_water_error_mm=float(np.max(np.abs(fields["OUT_WATER_ERROR"]))),
            )
            for name, values in fields.items():
                fields_history[name].append(values)
                if values.ndim == 1:
                    row[name] = float(np.average(values, weights=geometry["areas"]))
                else:
                    for layer, layer_values in enumerate(values):
                        row[f"{name}_{layer+1}"] = float(
                            np.average(layer_values, weights=geometry["areas"])
                        )
            records.append(row)
            head_history.append(head.copy())
            interface_history.append(volume.copy())
            if step % 5 == 0:
                print(
                    f"{args.case}: day {(step+1)*args.interval:g}; soil={row['OUT_SOIL_MOIST_3']:.4f} mm; exchange={row['OUT_GW_EXCHANGE']:.4f} mm",
                    flush=True,
                )
    finally:
        if runtime is not None:
            runtime.finalize()
    arrays = {name: np.array(values) for name, values in fields_history.items()}
    arrays.update(
        heads=np.array(head_history), interface_m3=np.array(interface_history)
    )
    if runtime is not None:
        budgets = extract_budgets(args.run_dir, arrays, args.interval)
        if budgets.max_cell_residual_m3.abs().max() > 1e-3:
            raise RuntimeError("groundwater cell budget exceeds 1e-3 m3 per window")
    table = pd.DataFrame(records)
    table.to_csv(args.run_dir / "timeseries.csv", index=False)
    np.savez_compressed(args.run_dir / "fields.npz", **arrays)
    if table.max_vic_water_error_mm.max() > 1e-7:
        raise RuntimeError("VIC water error exceeds 1e-7 mm per window")
    settings["elapsed_seconds"] = time.monotonic() - start_time
    settings["status"] = "completed"
    (args.run_dir / "provenance.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(
        f"completed {args.run_dir} in {settings['elapsed_seconds']:.1f} s", flush=True
    )


if __name__ == "__main__":
    main()
