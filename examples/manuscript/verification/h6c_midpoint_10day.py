#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
from datetime import date, timedelta
import csv
import math
import os
import subprocess
import time

import netCDF4 as nc
import numpy as np
from xmipy import XmiWrapper
from mpi4py import MPI


# ============================================================
# Configuration
# ============================================================

VIC_ROOT = Path(str(VIC_ROOT))

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

ROOT = Path(
    str(RESULT_ROOT)
)

SS_PER_M = float(
    os.environ.get(
        "H6C_SS_PER_M",
        "1e-5"
    )
)

SS_TAG = f"{SS_PER_M:.0e}"

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    ROOT /
    f"H6c_midpoint_10day_SS_{SS_TAG}_{STAMP}"
)

MF6_DIR = OUT / "mf6"
VIC_DIR = OUT / "vic"
GLOBAL_DIR = OUT / "globals"

for p in (
    OUT,
    MF6_DIR,
    VIC_DIR,
    GLOBAL_DIR,
):
    p.mkdir(
        parents=True,
        exist_ok=True
    )


START = date(1949, 1, 1)
NDAYS = 10

INITIAL_HEAD_M = -105.328

THICKNESS_M = 200.0

KA_SCALE = 0.001
EXCHANGE_LENGTH_M = 100.0
DRAIN_FRACTION = 1.0

HEAD_TOL_M = 1.0e-7
VOLUME_TOL_M3 = 1.0

MAX_ITER = 30


# ============================================================
# Domain
# ============================================================

with nc.Dataset(DOMAIN_FILE) as ds:

    area = np.ma.asarray(
        ds["area"][:],
        dtype=float
    )

    mask = (
        np.asarray(ds["mask"][:]) > 0
    )


valid_area = (
    mask &
    ~np.ma.getmaskarray(area)
)

TOTAL_AREA_M2 = float(
    np.sum(
        np.asarray(area)[valid_area]
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
print("===== H6c CONFIGURATION =====")
print("Ss                =", SS_PER_M, "1/m")
print("active VIC cells  =", int(np.sum(valid_area)))
print("total VIC area    =", TOTAL_AREA_M2, "m2")
print("storage coeff     =", STORAGE_COEFF_M2, "m2")


# ============================================================
# Find existing H6a/H5c reference
# ============================================================

refs = sorted(
    ROOT.glob(
        f"H5c_subdaily_coupling_SS_{SS_TAG}_*"
    ),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

if not refs:
    raise RuntimeError(
        "Could not find Ss=1e-5 H6a reference."
    )

REF = refs[0]


def read_csv(path):

    with path.open() as f:
        return list(
            csv.DictReader(f)
        )


hourly = read_csv(
    REF /
    "dt_01h" /
    "coupling_steps.csv"
)

explicit = read_csv(
    REF /
    "dt_24h" /
    "coupling_steps.csv"
)


if len(hourly) != 240:
    raise RuntimeError(
        f"Expected 240 hourly rows; got {len(hourly)}"
    )

if len(explicit) != 10:
    raise RuntimeError(
        f"Expected 10 daily explicit rows; got {len(explicit)}"
    )


reference_days = []

for d in range(NDAYS):

    rows = hourly[
        d * 24:
        (d + 1) * 24
    ]

    reference_days.append({
        "day":
            d + 1,

        "volume_m3":
            sum(
                float(r["vic_volume_m3"])
                for r in rows
            ),

        "head_after_m":
            float(
                rows[-1][
                    "mf6_head_after_m"
                ]
            ),

        "bottom_moist_mm":
            float(
                rows[-1][
                    "bottom_moist_mm"
                ]
            ),
    })


REFERENCE_TOTAL_VOLUME = sum(
    x["volume_m3"]
    for x in reference_days
)

REFERENCE_FINAL_HEAD = (
    reference_days[-1][
        "head_after_m"
    ]
)


EXPLICIT_TOTAL_VOLUME = sum(
    float(
        r["vic_volume_m3"]
    )
    for r in explicit
)

EXPLICIT_FINAL_HEAD = float(
    explicit[-1][
        "mf6_head_after_m"
    ]
)


print()
print("Reference directory:")
print(REF)

print()
print(
    "1h reference total volume =",
    REFERENCE_TOTAL_VOLUME,
    "m3"
)

print(
    "1h reference final head   =",
    REFERENCE_FINAL_HEAD,
    "m"
)

print(
    "24h explicit total volume =",
    EXPLICIT_TOTAL_VOLUME,
    "m3"
)


# ============================================================
# Global template
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

    "AGGFREQ",
}


lines = []

for line in BASE_GLOBAL.read_text().splitlines():

    s = line.strip()

    if (
        s and
        not s.startswith("#")
    ):

        if s.split()[0] in controlled:
            continue

    lines.append(line)


BASE_TEXT = (
    "\n".join(lines) +
    "\n"
)


# ============================================================
# MF6 model
# ============================================================

(MF6_DIR / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h6c.tdis
END TIMING

BEGIN MODELS
  GWF6 h6c.nam H6C
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h6c.ims H6C
END SOLUTIONGROUP
"""
)


(MF6_DIR / "h6c.tdis").write_text(
"""BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  10.0 10 1.0
END PERIODDATA
"""
)


(MF6_DIR / "h6c.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
)


(MF6_DIR / "h6c.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DIS6 h6c.dis DIS
  IC6  h6c.ic  IC
  NPF6 h6c.npf NPF
  STO6 h6c.sto STO
  API6 h6c.api VICAPI
  OC6  h6c.oc  OC
END PACKAGES
"""
)


(MF6_DIR / "h6c.dis").write_text(
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
    CONSTANT 0.0
  BOTM
    CONSTANT -200.0
END GRIDDATA
"""
)


(MF6_DIR / "h6c.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    CONSTANT {INITIAL_HEAD_M:.17g}
END GRIDDATA
"""
)


(MF6_DIR / "h6c.npf").write_text(
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


(MF6_DIR / "h6c.sto").write_text(
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


(MF6_DIR / "h6c.api").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND 1
END DIMENSIONS
"""
)


(MF6_DIR / "h6c.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h6c.hds
  BUDGET FILEOUT h6c.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
)


# ============================================================
# VIC evaluation
#
# IMPORTANT:
# Every iteration for a particular day starts from the SAME
# accepted beginning-of-day state.
# ============================================================

def evaluate_vic(
    day_index,
    current_day,
    boundary_head,
    init_state,
    iteration,
):

    run_dir = (
        VIC_DIR /
        f"day_{day_index:02d}" /
        f"iter_{iteration:02d}"
    )

    results = (
        run_dir /
        "results"
    )

    states = (
        run_dir /
        "state"
    )

    results.mkdir(
        parents=True,
        exist_ok=True
    )

    states.mkdir(
        parents=True,
        exist_ok=True
    )


    next_day = (
        current_day +
        timedelta(days=1)
    )


    global_file = (
        GLOBAL_DIR /
        f"day_{day_index:02d}_"
        f"iter_{iteration:02d}.global.txt"
    )


    extra = [
        "",
        f"STARTYEAR   {current_day.year}",
        f"STARTMONTH  {current_day.month}",
        f"STARTDAY    {current_day.day}",
        "STARTSEC    0",

        "NRECS       24",

        f"RESULT_DIR  {results}",

        "AGGFREQ     NHOURS 24",
    ]


    if init_state is not None:

        extra.append(
            f"INIT_STATE  {init_state}"
        )


    extra.extend([
        f"STATENAME   {states / 'state'}",

        f"STATEYEAR   {next_day.year}",
        f"STATEMONTH  {next_day.month}",
        f"STATEDAY    {next_day.day}",
        "STATESEC    0",

        "STATE_FORMAT NETCDF4_CLASSIC",
        "",
    ])


    global_file.write_text(
        BASE_TEXT +
        "\n".join(extra)
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
            str(DRAIN_FRACTION),

        "VIC_GW_HEAD_OFFSET_M":
            f"{boundary_head:.17g}",

        "VIC_GW_EXCHANGE_LENGTH_M":
            str(EXCHANGE_LENGTH_M),
    })


    with (
        run_dir /
        "stdout.log"
    ).open("w") as stdout, \
         (
             run_dir /
             "stderr.log"
         ).open("w") as stderr:

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
            stdout=stdout,
            stderr=stderr,
        )


    if proc.returncode != 0:

        raise RuntimeError(
            f"VIC failed day {day_index} "
            f"iteration {iteration}"
        )


    fluxes = sorted(
        results.glob(
            "fluxes*.nc"
        )
    )

    state_files = sorted(
        states.glob(
            "state*.nc"
        )
    )


    if len(fluxes) != 1:
        raise RuntimeError(
            "Expected one VIC flux file."
        )

    if len(state_files) != 1:
        raise RuntimeError(
            "Expected one VIC state file."
        )


    with nc.Dataset(
        fluxes[0]
    ) as ds:

        q = np.ma.asarray(
            ds[
                "OUT_GW_EXCHANGE"
            ][0]
        )

        soil = np.ma.asarray(
            ds[
                "OUT_SOIL_MOIST"
            ][0]
        )

        water_error = np.ma.asarray(
            ds[
                "OUT_WATER_ERROR"
            ][0]
        )


    valid = (
        valid_area &
        ~np.ma.getmaskarray(q)
    )

    av = np.asarray(
        area
    )[valid]

    qv = np.asarray(
        q
    )[valid]


    volume = float(
        np.sum(
            qv *
            1.0e-3 *
            av
        )
    )


    bottom = (
        soil[-1]
    )

    valid_bottom = (
        valid_area &
        ~np.ma.getmaskarray(
            bottom
        )
    )

    ab = np.asarray(
        area
    )[valid_bottom]

    bv = np.asarray(
        bottom
    )[valid_bottom]


    bottom_aw = float(
        np.sum(
            bv * ab
        ) /
        np.sum(ab)
    )


    wb = float(
        np.max(
            np.abs(
                water_error.compressed()
            )
        )
    )


    return {
        "boundary_head_m":
            boundary_head,

        "volume_m3":
            volume,

        "bottom_moist_mm":
            bottom_aw,

        "water_error_absmax_mm":
            wb,

        "state_file":
            state_files[0],

        "flux_file":
            fluxes[0],
    }


# ============================================================
# Persistent MF6
# ============================================================

mf6 = XmiWrapper(
    str(LIBMF6),
    working_directory=
        str(MF6_DIR)
)

mf6.initialize_mpi(MPI.COMM_SELF.py2f())


daily_rows = []
iteration_rows = []

accepted_state = None

cumulative_volume = 0.0


try:

    head = mf6.get_value_ptr(
        mf6.get_var_address(
            "X",
            "H6C"
        )
    )

    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H6C",
            "VICAPI"
        )
    )

    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H6C",
            "VICAPI"
        )
    )

    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H6C",
            "VICAPI"
        )
    )

    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H6C",
            "VICAPI"
        )
    )

    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H6C",
            "VICAPI"
        )
    )


    for day_idx in range(
        1,
        NDAYS + 1
    ):

        current_day = (
            START +
            timedelta(
                days=
                    day_idx - 1
            )
        )


        h_start = float(
            head[0]
        )


        print()
        print("=" * 76)

        print(
            f"H6c DAY {day_idx:02d} "
            f"{current_day}"
        )

        print("=" * 76)

        print(
            "MF6 start head =",
            h_start,
            "m"
        )


        # ----------------------------------------------------
        # Midpoint Picard iteration
        # ----------------------------------------------------

        midpoint_guess = (
            h_start
        )

        previous_volume = None

        accepted = None


        for iteration in range(
            1,
            MAX_ITER + 1
        ):

            trial = evaluate_vic(
                day_idx,
                current_day,
                midpoint_guess,

                # SAME beginning-of-day state every iteration.
                accepted_state,

                iteration,
            )


            volume = (
                trial[
                    "volume_m3"
                ]
            )


            predicted_end_head = (
                h_start +
                volume /
                STORAGE_COEFF_M2
            )


            updated_midpoint = (
                0.5 *
                (
                    h_start +
                    predicted_end_head
                )
            )


            head_change = abs(
                updated_midpoint -
                midpoint_guess
            )


            if previous_volume is None:

                volume_change = np.inf

            else:

                volume_change = abs(
                    volume -
                    previous_volume
                )


            iteration_rows.append({
                "day":
                    day_idx,

                "iteration":
                    iteration,

                "start_head_m":
                    h_start,

                "midpoint_guess_m":
                    midpoint_guess,

                "vic_volume_m3":
                    volume,

                "predicted_end_head_m":
                    predicted_end_head,

                "updated_midpoint_m":
                    updated_midpoint,

                "head_change_m":
                    head_change,

                "volume_change_m3":
                    volume_change,
            })


            print(
                f"iter={iteration:2d} "
                f"hmid={midpoint_guess:.9f} "
                f"V={volume:.3f} "
                f"hend={predicted_end_head:.9f} "
                f"|dh|={head_change:.3e} "
                f"|dV|={volume_change:.3e}"
            )


            accepted = trial


            if (
                head_change <=
                HEAD_TOL_M
                and
                volume_change <=
                VOLUME_TOL_M3
            ):

                break


            midpoint_guess = (
                updated_midpoint
            )

            previous_volume = (
                volume
            )


        else:

            raise RuntimeError(
                f"Picard iteration failed "
                f"on day {day_idx}"
            )


        # ----------------------------------------------------
        # ACCEPT final trial only now.
        #
        # All previous iterations are discarded.
        # ----------------------------------------------------

        accepted_volume = (
            accepted[
                "volume_m3"
            ]
        )


        predicted_end_head = (
            h_start +
            accepted_volume /
            STORAGE_COEFF_M2
        )


        # ----------------------------------------------------
        # Actual MF6 solve using accepted VIC flux
        # ----------------------------------------------------

        mf6.prepare_time_step(
            1.0
        )


        nbound[0] = 1
        nodelist[0] = 1
        hcof[0] = 0.0
        rhs[0] = -accepted_volume


        component = 1

        mf6.prepare_solve(
            component
        )


        converged = False
        mf6_iterations = 0


        for it in range(
            1,
            101
        ):

            nbound[0] = 1
            nodelist[0] = 1
            hcof[0] = 0.0
            rhs[0] = -accepted_volume


            converged = mf6.solve(
                component
            )

            mf6_iterations = it


            if converged:
                break


        if not converged:

            raise RuntimeError(
                f"MF6 failed day {day_idx}"
            )


        mf6.finalize_solve(
            component
        )

        mf6.finalize_time_step()


        actual_end_head = float(
            head[0]
        )

        api_flow = float(
            simvals[0]
        )


        mf6_storage_volume = (
            (
                actual_end_head -
                h_start
            ) *
            STORAGE_COEFF_M2
        )


        transfer_error = (
            mf6_storage_volume -
            accepted_volume
        )


        cumulative_volume += (
            accepted_volume
        )


        # This is the only VIC state that proceeds to tomorrow.
        accepted_state = (
            accepted[
                "state_file"
            ]
        )


        ref = (
            reference_days[
                day_idx - 1
            ]
        )

        exp = (
            explicit[
                day_idx - 1
            ]
        )


        daily_rows.append({
            "day":
                day_idx,

            "date":
                current_day.isoformat(),

            "picard_iterations":
                iteration,

            "mf6_head_start_m":
                h_start,

            "accepted_midpoint_head_m":
                accepted[
                    "boundary_head_m"
                ],

            "midpoint_volume_m3":
                accepted_volume,

            "mf6_head_end_m":
                actual_end_head,

            "predicted_head_end_m":
                predicted_end_head,

            "mf6_head_prediction_error_m":
                actual_end_head -
                predicted_end_head,

            "api_flow_m3_day":
                api_flow,

            "transfer_error_m3":
                transfer_error,

            "bottom_moist_mm":
                accepted[
                    "bottom_moist_mm"
                ],

            "vic_water_error_mm":
                accepted[
                    "water_error_absmax_mm"
                ],

            "reference_1h_volume_m3":
                ref[
                    "volume_m3"
                ],

            "volume_error_vs_1h_m3":
                accepted_volume -
                ref[
                    "volume_m3"
                ],

            "reference_1h_head_m":
                ref[
                    "head_after_m"
                ],

            "head_error_vs_1h_m":
                actual_end_head -
                ref[
                    "head_after_m"
                ],

            "explicit_24h_volume_m3":
                float(
                    exp[
                        "vic_volume_m3"
                    ]
                ),
        })


        print(
            "accepted midpoint =",
            accepted[
                "boundary_head_m"
            ],
            "m"
        )

        print(
            "accepted volume   =",
            accepted_volume,
            "m3"
        )

        print(
            "MF6 end head      =",
            actual_end_head,
            "m"
        )

        print(
            "1h ref volume     =",
            ref[
                "volume_m3"
            ],
            "m3"
        )

        print(
            "dV vs 1h         =",
            accepted_volume -
            ref[
                "volume_m3"
            ],
            "m3"
        )


    FINAL_HEAD = float(
        head[0]
    )


finally:

    mf6.finalize()


# ============================================================
# Save tables
# ============================================================

with (
    OUT /
    "iterations.csv"
).open(
    "w",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=
            iteration_rows[0].keys()
    )

    w.writeheader()
    w.writerows(
        iteration_rows
    )


with (
    OUT /
    "daily.csv"
).open(
    "w",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=
            daily_rows[0].keys()
    )

    w.writeheader()
    w.writerows(
        daily_rows
    )


# ============================================================
# Final comparison
# ============================================================

MIDPOINT_VOLUME = (
    cumulative_volume
)

MIDPOINT_HEAD = (
    FINAL_HEAD
)


explicit_volume_error = (
    EXPLICIT_TOTAL_VOLUME -
    REFERENCE_TOTAL_VOLUME
)

midpoint_volume_error = (
    MIDPOINT_VOLUME -
    REFERENCE_TOTAL_VOLUME
)


explicit_percent = (
    100.0 *
    explicit_volume_error /
    abs(
        REFERENCE_TOTAL_VOLUME
    )
)

midpoint_percent = (
    100.0 *
    midpoint_volume_error /
    abs(
        REFERENCE_TOTAL_VOLUME
    )
)


explicit_head_error = (
    EXPLICIT_FINAL_HEAD -
    REFERENCE_FINAL_HEAD
)

midpoint_head_error = (
    MIDPOINT_HEAD -
    REFERENCE_FINAL_HEAD
)


improvement_factor = (
    abs(
        explicit_volume_error
    ) /
    abs(
        midpoint_volume_error
    )
    if midpoint_volume_error != 0.0
    else np.inf
)


max_picard_iterations = max(
    row[
        "picard_iterations"
    ]
    for row in daily_rows
)


mf6_storage_total = (
    (
        MIDPOINT_HEAD -
        INITIAL_HEAD_M
    ) *
    STORAGE_COEFF_M2
)

conservation_error = (
    mf6_storage_total -
    MIDPOINT_VOLUME
)


summary = {
    "Ss_per_m":
        SS_PER_M,

    "reference_1h_volume_m3":
        REFERENCE_TOTAL_VOLUME,

    "explicit_24h_volume_m3":
        EXPLICIT_TOTAL_VOLUME,

    "midpoint_24h_volume_m3":
        MIDPOINT_VOLUME,

    "explicit_volume_error_m3":
        explicit_volume_error,

    "midpoint_volume_error_m3":
        midpoint_volume_error,

    "explicit_volume_error_percent":
        explicit_percent,

    "midpoint_volume_error_percent":
        midpoint_percent,

    "midpoint_improvement_factor":
        improvement_factor,

    "reference_1h_final_head_m":
        REFERENCE_FINAL_HEAD,

    "explicit_24h_final_head_m":
        EXPLICIT_FINAL_HEAD,

    "midpoint_final_head_m":
        MIDPOINT_HEAD,

    "explicit_head_error_m":
        explicit_head_error,

    "midpoint_head_error_m":
        midpoint_head_error,

    "max_picard_iterations":
        max_picard_iterations,

    "cross_model_conservation_error_m3":
        conservation_error,

    "relative_conservation_error":
        conservation_error /
        MIDPOINT_VOLUME,
}


with (
    OUT /
    "summary.csv"
).open(
    "w",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=
            summary.keys()
    )

    w.writeheader()
    w.writerow(
        summary
    )


print()
print("=" * 100)
print("H6c 10-DAY MIDPOINT-PICARD COMPARISON")
print("=" * 100)

print(
    f"{'method':>18s} "
    f"{'cum volume':>18s} "
    f"{'dV vs 1h':>14s} "
    f"{'% error':>12s} "
    f"{'final head':>15s} "
    f"{'dh vs 1h':>13s}"
)


print(
    f"{'1h explicit':>18s} "
    f"{REFERENCE_TOTAL_VOLUME:18.3f} "
    f"{0.0:14.3f} "
    f"{0.0:12.6f} "
    f"{REFERENCE_FINAL_HEAD:15.9f} "
    f"{0.0:13.6g}"
)

print(
    f"{'24h explicit':>18s} "
    f"{EXPLICIT_TOTAL_VOLUME:18.3f} "
    f"{explicit_volume_error:14.3f} "
    f"{explicit_percent:12.6f} "
    f"{EXPLICIT_FINAL_HEAD:15.9f} "
    f"{explicit_head_error:13.6g}"
)

print(
    f"{'24h midpoint':>18s} "
    f"{MIDPOINT_VOLUME:18.3f} "
    f"{midpoint_volume_error:14.3f} "
    f"{midpoint_percent:12.6f} "
    f"{MIDPOINT_HEAD:15.9f} "
    f"{midpoint_head_error:13.6g}"
)


print()
print(
    "midpoint improvement factor =",
    improvement_factor
)

print(
    "max Picard iterations       =",
    max_picard_iterations
)

print(
    "cross-model volume error    =",
    conservation_error,
    "m3"
)

print()
print("Saved:")
print(OUT / "iterations.csv")
print(OUT / "daily.csv")
print(OUT / "summary.csv")

print()
print("Experiment directory:")
print(OUT)
