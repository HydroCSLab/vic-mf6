#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
from datetime import date, timedelta
import csv
import math
import os
import shutil
import subprocess
import time

import netCDF4 as nc
import numpy as np
from xmipy import XmiWrapper
from mpi4py import MPI


# ============================================================
# Paths
# ============================================================

VIC_ROOT = Path(
    str(VIC_ROOT)
)

VIC_EXE = (
    VIC_ROOT /
    "vic/drivers/image/vic_image.exe"
)

PARAMDIR = Path(
    str(SAMPLE_ROOT / 'parameters')
)

BASE_GLOBAL = (
    PARAMDIR /
    "Stehekin_image_test.global.txt"
)

DOMAIN_FILE = (
    PARAMDIR /
    "domain.stehekin.20151028.nc"
)

LIBMF6 = Path(
    str(MF6_LIBRARY)
)

EXPROOT = Path(
    str(RESULT_ROOT)
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    EXPROOT /
    f"H4_coupling_interval_{STAMP}"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experiment
# ============================================================

START = date(1949, 1, 1)
TOTAL_DAYS = 10

INTERVALS = [
    1,
    2,
    5,
    10,
]

INITIAL_HEAD_M = -105.328

EXCHANGE_LENGTH_M = 100.0
KA_SCALE = 0.001
DRAIN_FRACTION = 1.0

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)

SS_PER_M = 0.001


# ============================================================
# VIC domain area
# ============================================================

with nc.Dataset(DOMAIN_FILE) as ds:

    area_var = ds["area"]

    area = np.ma.asarray(
        area_var[:],
        dtype=float
    )

    domain_mask = (
        np.asarray(ds["mask"][:]) > 0
    )

    units = (
        getattr(area_var, "units", "")
        .strip()
        .lower()
        .replace(" ", "")
    )


if units in (
    "m2",
    "m^2",
    "m**2",
):

    area_m2 = area

elif units in (
    "km2",
    "km^2",
    "km**2",
):

    area_m2 = (
        area *
        1.0e6
    )

else:

    raise RuntimeError(
        f"Unsupported area units: {units!r}"
    )


valid_area = (
    domain_mask &
    ~np.ma.getmaskarray(
        area_m2
    )
)

TOTAL_AREA_M2 = float(
    np.sum(
        np.asarray(area_m2)[
            valid_area
        ]
    )
)

CELL_WIDTH_M = math.sqrt(
    TOTAL_AREA_M2
)

STORAGE_COEFF_M2 = (
    SS_PER_M *
    TOTAL_AREA_M2 *
    THICKNESS_M
)


print()
print("===== H4 DOMAIN =====")

print(
    "active VIC cells =",
    int(np.sum(valid_area))
)

print(
    "total VIC area   =",
    TOTAL_AREA_M2,
    "m2"
)

print(
    "MF6 storage coeff =",
    STORAGE_COEFF_M2,
    "m2"
)


# ============================================================
# Provenance
# ============================================================

def capture(cmd):
    from manuscript_paths import source_provenance
    return source_provenance(cmd)


(OUT / "vic_commit.txt").write_text(
    capture([
        "git",
        "rev-parse",
        "HEAD",
    ])
)

(OUT / "vic_status.txt").write_text(
    capture([
        "git",
        "status",
        "--short",
    ])
)

(OUT / "vic_diff.txt").write_text(
    capture([
        "git",
        "diff",
    ])
)

shutil.copy2(
    BASE_GLOBAL,
    OUT / "canonical_global.txt"
)


# ============================================================
# Prepare VIC global template
# ============================================================

controlled = {
    "STARTYEAR",
    "STARTMONTH",
    "STARTDAY",
    "STARTSEC",

    "ENDYEAR",
    "ENDMONTH",
    "ENDDAY",

    "NRECS",

    "RESULT_DIR",

    "INIT_STATE",

    "STATENAME",
    "STATEYEAR",
    "STATEMONTH",
    "STATEDAY",
    "STATESEC",
    "STATE_FORMAT",
}


base_lines = []

for line in BASE_GLOBAL.read_text().splitlines():

    stripped = line.strip()

    if (
        stripped and
        not stripped.startswith("#")
    ):

        key = stripped.split()[0]

        if key in controlled:
            continue

    base_lines.append(line)


clean_global = (
    "\n".join(base_lines) +
    "\n"
)


def make_global(
    path,
    start_day,
    end_day,
    state_time,
    result_dir,
    state_prefix,
    init_state,
):

    lines = [
        "",
        "# H4 coupling-interval experiment",

        f"STARTYEAR    {start_day.year}",
        f"STARTMONTH   {start_day.month}",
        f"STARTDAY     {start_day.day}",
        "STARTSEC     0",

        f"ENDYEAR      {end_day.year}",
        f"ENDMONTH     {end_day.month}",
        f"ENDDAY       {end_day.day}",

        f"RESULT_DIR   {result_dir}",
    ]

    if init_state is not None:

        lines.append(
            f"INIT_STATE   {init_state}"
        )

    lines.extend([
        f"STATENAME    {state_prefix}",

        f"STATEYEAR    {state_time.year}",
        f"STATEMONTH   {state_time.month}",
        f"STATEDAY     {state_time.day}",
        "STATESEC     0",

        "STATE_FORMAT NETCDF4_CLASSIC",
        "",
    ])


    path.write_text(
        clean_global +
        "\n".join(lines)
    )


def find_one(
    directory,
    pattern,
):

    files = sorted(
        directory.glob(pattern)
    )

    if len(files) != 1:

        raise RuntimeError(
            f"Expected one {pattern} in "
            f"{directory}; got {files}"
        )

    return files[0]


# ============================================================
# Integrate VIC groundwater exchange over one chunk
# ============================================================

def integrate_vic_flux(
    flux_file
):

    with nc.Dataset(
        flux_file
    ) as ds:

        q = np.ma.asarray(
            ds[
                "OUT_GW_EXCHANGE"
            ][:]
        )

        soil = np.ma.asarray(
            ds[
                "OUT_SOIL_MOIST"
            ][:]
        )

        runoff = np.ma.asarray(
            ds[
                "OUT_RUNOFF"
            ][:]
        )

        water_error = np.ma.asarray(
            ds[
                "OUT_WATER_ERROR"
            ][:]
        )


    total_volume = 0.0

    daily_volumes = []

    daily_aw_q = []


    for t in range(q.shape[0]):

        qt = q[t]

        valid = (
            valid_area &
            ~np.ma.getmaskarray(qt)
        )

        av = np.asarray(
            area_m2
        )[valid]

        qv = np.asarray(
            qt
        )[valid]


        volume = float(
            np.sum(
                qv *
                1.0e-3 *
                av
            )
        )

        total_volume += volume

        daily_volumes.append(
            volume
        )

        daily_aw_q.append(
            volume /
            TOTAL_AREA_M2 *
            1000.0
        )


    final_bottom = (
        soil[-1, -1]
    )

    valid_bottom = (
        valid_area &
        ~np.ma.getmaskarray(
            final_bottom
        )
    )

    av = np.asarray(
        area_m2
    )[valid_bottom]

    bv = np.asarray(
        final_bottom
    )[valid_bottom]


    bottom_aw = float(
        np.sum(
            bv * av
        ) /
        np.sum(av)
    )


    final_runoff = runoff[-1]

    valid_runoff = (
        valid_area &
        ~np.ma.getmaskarray(
            final_runoff
        )
    )

    av = np.asarray(
        area_m2
    )[valid_runoff]

    rv = np.asarray(
        final_runoff
    )[valid_runoff]


    runoff_aw = float(
        np.sum(
            rv * av
        ) /
        np.sum(av)
    )


    wb_absmax = float(
        np.max(
            np.abs(
                water_error.compressed()
            )
        )
    )


    return {
        "chunk_volume_m3":
            total_volume,

        "daily_volumes_m3":
            daily_volumes,

        "daily_aw_q_mm":
            daily_aw_q,

        "final_bottom_moist_mm":
            bottom_aw,

        "final_runoff_mm":
            runoff_aw,

        "water_error_absmax_mm":
            wb_absmax,
    }


# ============================================================
# Write MF6 model for one coupling interval
# ============================================================

def write_mf6(
    work,
    interval_days,
):

    nsteps = (
        TOTAL_DAYS //
        interval_days
    )


    (work / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h4.tdis
END TIMING

BEGIN MODELS
  GWF6 h4.nam H4
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h4.ims H4
END SOLUTIONGROUP
"""
    )


    (work / "h4.tdis").write_text(
f"""BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  {TOTAL_DAYS}.0 {nsteps} 1.0
END PERIODDATA
"""
    )


    (work / "h4.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
    )


    (work / "h4.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DIS6 h4.dis DIS
  IC6  h4.ic  IC
  NPF6 h4.npf NPF
  STO6 h4.sto STO
  API6 h4.api VICAPI
  OC6  h4.oc  OC
END PACKAGES
"""
    )


    (work / "h4.dis").write_text(
f"""BEGIN OPTIONS
  LENGTH_UNITS METERS
END OPTIONS

BEGIN DIMENSIONS
  NLAY 1
  NROW 1
  NCOL 1
END DIMENSIONS

BEGIN GRIDDATA
  DELR
    CONSTANT {CELL_WIDTH_M:.17g}
  DELC
    CONSTANT {CELL_WIDTH_M:.17g}
  TOP
    CONSTANT {TOP_M:.17g}
  BOTM
    CONSTANT {BOT_M:.17g}
END GRIDDATA
"""
    )


    (work / "h4.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    CONSTANT {INITIAL_HEAD_M:.17g}
END GRIDDATA
"""
    )


    (work / "h4.npf").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICELLTYPE
    CONSTANT 0
  K
    CONSTANT 1.0
END GRIDDATA
"""
    )


    (work / "h4.sto").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICONVERT
    CONSTANT 0
  SS
    CONSTANT {SS_PER_M:.17g}
  SY
    CONSTANT 0.20
END GRIDDATA

BEGIN PERIOD 1
  TRANSIENT
END PERIOD
"""
    )


    (work / "h4.api").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND 1
END DIMENSIONS
"""
    )


    (work / "h4.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h4.hds
  BUDGET FILEOUT h4.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
    )


# ============================================================
# Run one H4 experiment
# ============================================================

def run_interval(
    interval_days
):

    tag = (
        f"dt_{interval_days:02d}d"
    )

    root = (
        OUT /
        tag
    )

    vic_root = (
        root /
        "vic"
    )

    mf6_work = (
        root /
        "mf6"
    )

    globals_dir = (
        root /
        "globals"
    )

    logs_dir = (
        root /
        "logs"
    )


    for p in (
        vic_root,
        mf6_work,
        globals_dir,
        logs_dir,
    ):

        p.mkdir(
            parents=True,
            exist_ok=True
        )


    write_mf6(
        mf6_work,
        interval_days
    )


    mf6 = XmiWrapper(
        str(LIBMF6),
        working_directory=
            str(mf6_work)
    )

    mf6.initialize_mpi(MPI.COMM_SELF.py2f())


    chunk_rows = []

    cumulative_vic_volume = 0.0

    init_state = None

    final_bottom = np.nan
    final_runoff = np.nan
    max_wb_error = 0.0


    try:

        head = mf6.get_value_ptr(
            mf6.get_var_address(
                "X",
                "H4"
            )
        )

        nbound = mf6.get_value_ptr(
            mf6.get_var_address(
                "NBOUND",
                "H4",
                "VICAPI"
            )
        )

        nodelist = mf6.get_value_ptr(
            mf6.get_var_address(
                "NODELIST",
                "H4",
                "VICAPI"
            )
        )

        rhs = mf6.get_value_ptr(
            mf6.get_var_address(
                "RHS",
                "H4",
                "VICAPI"
            )
        )

        hcof = mf6.get_value_ptr(
            mf6.get_var_address(
                "HCOF",
                "H4",
                "VICAPI"
            )
        )

        simvals = mf6.get_value_ptr(
            mf6.get_var_address(
                "SIMVALS",
                "H4",
                "VICAPI"
            )
        )


        nchunks = (
            TOTAL_DAYS //
            interval_days
        )


        for k in range(nchunks):

            chunk_start = (
                START +
                timedelta(
                    days=
                        k *
                        interval_days
                )
            )

            chunk_end = (
                chunk_start +
                timedelta(
                    days=
                        interval_days -
                        1
                )
            )

            state_time = (
                chunk_start +
                timedelta(
                    days=
                        interval_days
                )
            )


            head_before = float(
                head[0]
            )


            print()
            print(
                "=" * 72
            )

            print(
                f"{tag} chunk {k + 1}/{nchunks}"
            )

            print(
                chunk_start,
                "through",
                chunk_end
            )

            print(
                "head before =",
                head_before,
                "m"
            )


            chunk_dir = (
                vic_root /
                f"chunk_{k + 1:02d}"
            )

            result_dir = (
                chunk_dir /
                "results"
            )

            state_dir = (
                chunk_dir /
                "state"
            )

            result_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            state_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            state_prefix = (
                state_dir /
                "state"
            )

            global_file = (
                globals_dir /
                f"chunk_{k + 1:02d}.global.txt"
            )


            make_global(
                global_file,
                chunk_start,
                chunk_end,
                state_time,
                result_dir,
                state_prefix,
                init_state,
            )


            env = os.environ.copy()

            env.update({
                "OMP_NUM_THREADS":
                    "1",

                "VIC_GW_FORMULATION":
                    "head",

                "VIC_GW_REFERENCE_DEPTH":
                    "base",

                "VIC_GW_TEST_GAP_M":
                    "105.328",

                "VIC_GW_TEST_KA_SCALE":
                    str(KA_SCALE),

                "VIC_GW_MAX_DRAIN_FRACTION":
                    str(
                        DRAIN_FRACTION
                    ),

                "VIC_GW_HEAD_OFFSET_M":
                    f"{head_before:.17g}",

                "VIC_GW_EXCHANGE_LENGTH_M":
                    str(
                        EXCHANGE_LENGTH_M
                    ),
            })


            stdout = (
                logs_dir /
                f"chunk_{k + 1:02d}.stdout.log"
            )

            stderr = (
                logs_dir /
                f"chunk_{k + 1:02d}.stderr.log"
            )


            with stdout.open("w") as out, \
                 stderr.open("w") as err:

                proc = subprocess.run(
                    [
                        "mpirun",
                        "-n",
                        "1",
                        str(VIC_EXE),
                        "-g",
                        str(global_file),
                    ],
                    cwd=PARAMDIR,
                    env=env,
                    stdout=out,
                    stderr=err,
                )


            if proc.returncode != 0:

                raise RuntimeError(
                    f"VIC failed for "
                    f"{tag} chunk "
                    f"{k + 1}; see "
                    f"{stderr}"
                )


            flux_file = find_one(
                result_dir,
                "fluxes*.nc"
            )

            state_file = find_one(
                state_dir,
                "state*.nc"
            )


            info = integrate_vic_flux(
                flux_file
            )


            chunk_volume = (
                info[
                    "chunk_volume_m3"
                ]
            )

            Q_average = (
                chunk_volume /
                interval_days
            )


            cumulative_vic_volume += (
                chunk_volume
            )

            final_bottom = (
                info[
                    "final_bottom_moist_mm"
                ]
            )

            final_runoff = (
                info[
                    "final_runoff_mm"
                ]
            )

            max_wb_error = max(
                max_wb_error,
                info[
                    "water_error_absmax_mm"
                ]
            )


            print(
                "chunk VIC volume =",
                chunk_volume,
                "m3"
            )

            print(
                "average MF6 Q    =",
                Q_average,
                "m3/day"
            )


            # =================================================
            # MF6 timestep
            #
            # TDIS has been constructed so the actual MF6
            # timestep equals interval_days.
            # =================================================

            mf6.prepare_time_step(
                float(
                    interval_days
                )
            )


            nbound[0] = 1
            nodelist[0] = 1
            hcof[0] = 0.0
            rhs[0] = -Q_average


            component = 1

            mf6.prepare_solve(
                component
            )


            converged = False
            iterations = 0


            for iteration in range(
                1,
                101
            ):

                nbound[0] = 1
                nodelist[0] = 1
                hcof[0] = 0.0
                rhs[0] = -Q_average


                converged = mf6.solve(
                    component
                )

                iterations = iteration

                if converged:
                    break


            if not converged:

                raise RuntimeError(
                    f"MF6 failed for "
                    f"{tag} chunk "
                    f"{k + 1}"
                )


            mf6.finalize_solve(
                component
            )

            mf6.finalize_time_step()


            head_after = float(
                head[0]
            )

            api_Q = float(
                simvals[0]
            )


            dh = (
                head_after -
                head_before
            )

            storage_volume = (
                dh *
                STORAGE_COEFF_M2
            )

            volume_error = (
                storage_volume -
                chunk_volume
            )


            print(
                "head after       =",
                head_after,
                "m"
            )

            print(
                "dh               =",
                dh,
                "m"
            )

            print(
                "volume error     =",
                volume_error,
                "m3"
            )


            chunk_rows.append({
                "interval_days":
                    interval_days,

                "chunk":
                    k + 1,

                "start_date":
                    chunk_start.isoformat(),

                "end_date":
                    chunk_end.isoformat(),

                "mf6_head_before_m":
                    head_before,

                "vic_chunk_volume_m3":
                    chunk_volume,

                "mf6_average_Q_m3_day":
                    Q_average,

                "mf6_simvals_m3_day":
                    api_Q,

                "mf6_iterations":
                    iterations,

                "mf6_head_after_m":
                    head_after,

                "mf6_dh_m":
                    dh,

                "mf6_storage_volume_m3":
                    storage_volume,

                "volume_error_m3":
                    volume_error,

                "vic_water_error_absmax_mm":
                    info[
                        "water_error_absmax_mm"
                    ],
            })


            if not np.isclose(
                api_Q,
                Q_average,
                rtol=1.0e-12,
                atol=1.0e-7,
            ):

                raise RuntimeError(
                    "MF6 API-flow mismatch."
                )


            if not np.isclose(
                storage_volume,
                chunk_volume,
                rtol=1.0e-9,
                atol=1.0e-3,
            ):

                raise RuntimeError(
                    "Cross-model volume mismatch."
                )


            init_state = (
                state_file
            )


        final_head = float(
            head[0]
        )


    finally:

        mf6.finalize()


    analytical_storage = (
        (
            final_head -
            INITIAL_HEAD_M
        ) *
        STORAGE_COEFF_M2
    )


    conservation_error = (
        analytical_storage -
        cumulative_vic_volume
    )


    with (
        root /
        "chunks.csv"
    ).open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                chunk_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            chunk_rows
        )


    return {
        "interval_days":
            interval_days,

        "n_coupling_exchanges":
            TOTAL_DAYS //
            interval_days,

        "final_mf6_head_m":
            final_head,

        "mf6_head_change_m":
            final_head -
            INITIAL_HEAD_M,

        "cumulative_vic_volume_m3":
            cumulative_vic_volume,

        "cumulative_equivalent_mm":
            cumulative_vic_volume /
            TOTAL_AREA_M2 *
            1000.0,

        "mf6_storage_change_m3":
            analytical_storage,

        "conservation_error_m3":
            conservation_error,

        "relative_conservation_error":
            conservation_error /
            cumulative_vic_volume
            if cumulative_vic_volume != 0
            else 0.0,

        "final_bottom_moist_mm":
            final_bottom,

        "final_runoff_mm":
            final_runoff,

        "max_vic_water_error_mm":
            max_wb_error,

        "max_chunk_transfer_error_m3":
            max(
                abs(
                    row[
                        "volume_error_m3"
                    ]
                )
                for row in chunk_rows
            ),
    }


# ============================================================
# Run all coupling intervals
# ============================================================

results = []


for interval in INTERVALS:

    print()
    print("#" * 72)

    print(
        "H4 COUPLING INTERVAL:",
        interval,
        "day(s)"
    )

    print("#" * 72)


    results.append(
        run_interval(
            interval
        )
    )


# ============================================================
# Compare against one-day reference
# ============================================================

reference = next(
    row
    for row in results
    if row["interval_days"] == 1
)


for row in results:

    row[
        "final_head_diff_vs_1d_m"
    ] = (
        row[
            "final_mf6_head_m"
        ] -
        reference[
            "final_mf6_head_m"
        ]
    )


    row[
        "volume_diff_vs_1d_m3"
    ] = (
        row[
            "cumulative_vic_volume_m3"
        ] -
        reference[
            "cumulative_vic_volume_m3"
        ]
    )


    row[
        "volume_diff_vs_1d_percent"
    ] = (
        100.0 *
        row[
            "volume_diff_vs_1d_m3"
        ] /
        abs(
            reference[
                "cumulative_vic_volume_m3"
            ]
        )
    )


    row[
        "bottom_moist_diff_vs_1d_mm"
    ] = (
        row[
            "final_bottom_moist_mm"
        ] -
        reference[
            "final_bottom_moist_mm"
        ]
    )


# ============================================================
# Save
# ============================================================

summary_file = (
    OUT /
    "summary.csv"
)

with summary_file.open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            results[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        results
    )


# ============================================================
# Print compact scientific summary
# ============================================================

print()
print("=" * 105)
print("H4 COUPLING-INTERVAL SUMMARY")
print("=" * 105)

print(
    f"{'dt':>5s} "
    f"{'nexch':>6s} "
    f"{'final head':>16s} "
    f"{'cum volume':>18s} "
    f"{'dV vs 1d':>14s} "
    f"{'% vs 1d':>11s} "
    f"{'dh vs 1d':>14s} "
    f"{'dS3 vs 1d':>13s} "
    f"{'cons err':>12s}"
)


for row in results:

    print(
        f"{row['interval_days']:5d} "
        f"{row['n_coupling_exchanges']:6d} "
        f"{row['final_mf6_head_m']:16.9f} "
        f"{row['cumulative_vic_volume_m3']:18.3f} "
        f"{row['volume_diff_vs_1d_m3']:14.3f} "
        f"{row['volume_diff_vs_1d_percent']:11.6f} "
        f"{row['final_head_diff_vs_1d_m']:14.8e} "
        f"{row['bottom_moist_diff_vs_1d_mm']:13.8e} "
        f"{row['conservation_error_m3']:12.3e}"
    )


print()
print("Reference 1-day final head:")
print(
    reference[
        "final_mf6_head_m"
    ]
)

print()
print("Reference 1-day cumulative volume:")
print(
    reference[
        "cumulative_vic_volume_m3"
    ],
    "m3"
)

print()
print("Saved:")
print(summary_file)
print()
print("Experiment directory:")
print(OUT)
