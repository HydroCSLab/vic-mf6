#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
from datetime import date, timedelta
import csv
import os
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

LIBMF6 = Path(
    str(MF6_LIBRARY)
)

ROOT = Path(
    str(RESULT_ROOT)
)

H7B = (
    ROOT /
    "H7b_stehekin_real_mapper"
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    ROOT /
    f"H7e_spatial_10day_closed_loop_{STAMP}"
)

MF6_DIR = OUT / "mf6"
VIC_ROOT_OUT = OUT / "vic"

MF6_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIC_ROOT_OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experiment configuration
# ============================================================

START = date(
    1949,
    1,
    1,
)

NDAYS = 10

SS_PER_M = 1.0e-5

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)

DT_DAYS = 1.0

KA_SCALE = 0.001

EXCHANGE_LENGTH_M = 100.0

DRAIN_FRACTION = 1.0


# ============================================================
# Read H7b VIC cells
# ============================================================

with (
    H7B /
    "vic_cells.csv"
).open() as f:

    vic_rows = list(
        csv.DictReader(f)
    )


vic_rows.sort(
    key=lambda r:
        int(r["vic_id"])
)


if len(vic_rows) != 16:

    raise RuntimeError(
        f"Expected 16 VIC cells; "
        f"found {len(vic_rows)}"
    )


NV = len(
    vic_rows
)


vic_row = np.array(
    [
        int(r["row"])
        for r in vic_rows
    ],
    dtype=int,
)

vic_col = np.array(
    [
        int(r["col"])
        for r in vic_rows
    ],
    dtype=int,
)

vic_lat = np.array(
    [
        float(r["lat"])
        for r in vic_rows
    ],
    dtype=float,
)

vic_lon = np.array(
    [
        float(r["lon"])
        for r in vic_rows
    ],
    dtype=float,
)

vic_area_m2 = np.array(
    [
        float(r["area_m2"])
        for r in vic_rows
    ],
    dtype=float,
)


# ============================================================
# Read H7b MF6 cells
# ============================================================

with (
    H7B /
    "mf6_cells.csv"
).open() as f:

    mf6_rows = list(
        csv.DictReader(f)
    )


mf6_rows.sort(
    key=lambda r:
        int(r["mf6_id"])
)


if len(mf6_rows) != 12:

    raise RuntimeError(
        f"Expected 12 MF6 cells; "
        f"found {len(mf6_rows)}"
    )


NM = len(
    mf6_rows
)


mf6_area_m2 = np.array(
    [
        float(
            r["rectangle_area_m2"]
        )
        for r in mf6_rows
    ],
    dtype=float,
)


mf6_coupled_area_m2 = np.array(
    [
        float(
            r["coupled_area_m2"]
        )
        for r in mf6_rows
    ],
    dtype=float,
)


mf6_initial_head_m = np.array(
    [
        float(
            r["head_m"]
        )
        for r in mf6_rows
    ],
    dtype=float,
)


# ============================================================
# Overlap matrix
# ============================================================

A = np.zeros(
    (
        NV,
        NM
    ),
    dtype=float,
)


with (
    H7B /
    "exchange_table.csv"
).open() as f:

    overlaps = list(
        csv.DictReader(f)
    )


for r in overlaps:

    i = int(
        r["vic_id"]
    )

    j = int(
        r["mf6_id"]
    )

    A[i, j] = float(
        r["overlap_area_m2"]
    )


vic_covered_m2 = A.sum(
    axis=1
)


if not np.allclose(
    vic_covered_m2,
    vic_area_m2,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "Overlap matrix does not completely "
        "cover the active VIC cells."
    )


storage_coeff_m2 = (
    SS_PER_M *
    mf6_area_m2 *
    THICKNESS_M
)


print()
print("===== H7e DOMAIN =====")

print(
    "VIC cells       =",
    NV
)

print(
    "MF6 cells       =",
    NM
)

print(
    "overlap records =",
    int(
        np.count_nonzero(A)
    )
)

print(
    "VIC area        =",
    vic_area_m2.sum(),
    "m2"
)

print(
    "MF6 full area   =",
    mf6_area_m2.sum(),
    "m2"
)

print(
    "initial MF6 head range =",
    mf6_initial_head_m.min(),
    mf6_initial_head_m.max(),
    "m"
)


# ============================================================
# Optional H7d one-day regression reference
# ============================================================

h7d_candidates = sorted(
    ROOT.glob(
        "H7d_spatial_closed_loop_*"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)


H7D_REF = (
    h7d_candidates[0]
    if h7d_candidates
    else None
)


h7d_vic_q = None
h7d_mf6_volume = None
h7d_mf6_head = None
h7d_vic_return_head = None


if H7D_REF is not None:

    with (
        H7D_REF /
        "vic_cells.csv"
    ).open() as f:

        rows = list(
            csv.DictReader(f)
        )

    rows.sort(
        key=lambda r:
            int(r["vic_id"])
    )

    h7d_vic_q = np.array(
        [
            float(
                r["gw_exchange_mm"]
            )
            for r in rows
        ]
    )

    h7d_vic_return_head = np.array(
        [
            float(
                r[
                    "final_mapped_mf6_head_m"
                ]
            )
            for r in rows
        ]
    )


    with (
        H7D_REF /
        "mf6_cells.csv"
    ).open() as f:

        rows = list(
            csv.DictReader(f)
        )

    rows.sort(
        key=lambda r:
            int(r["mf6_id"])
    )

    h7d_mf6_volume = np.array(
        [
            float(
                r["mapped_net_volume_m3"]
            )
            for r in rows
        ]
    )

    h7d_mf6_head = np.array(
        [
            float(
                r["actual_head_final_m"]
            )
            for r in rows
        ]
    )


    print()
    print(
        "H7d regression reference =",
        H7D_REF
    )


# ============================================================
# Prepare base VIC global text
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


base_lines = []


for line in (
    BASE_GLOBAL
    .read_text()
    .splitlines()
):

    s = line.strip()

    if (
        s and
        not s.startswith("#")
    ):

        if s.split()[0] in controlled:
            continue

    base_lines.append(
        line
    )


BASE_TEXT = (
    "\n".join(
        base_lines
    ) +
    "\n"
)


# ============================================================
# VIC one-day runner
# ============================================================

def run_vic_day(
    day_index,
    current_day,
    vic_head_m,
    init_state,
):

    run_dir = (
        VIC_ROOT_OUT /
        f"day_{day_index:02d}"
    )

    result_dir = (
        run_dir /
        "results"
    )

    state_dir = (
        run_dir /
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


    # --------------------------------------------------------
    # Write current spatial groundwater-head field
    # --------------------------------------------------------

    head_file = (
        run_dir /
        "vic_gw_head.txt"
    )


    with head_file.open(
        "w"
    ) as f:

        f.write(
            "# latitude longitude head_offset_m\n"
        )

        for i in range(NV):

            f.write(
                f"{vic_lat[i]:.17g} "
                f"{vic_lon[i]:.17g} "
                f"{vic_head_m[i]:.17g}\n"
            )


    # --------------------------------------------------------
    # Global file
    # --------------------------------------------------------

    next_day = (
        current_day +
        timedelta(
            days=1
        )
    )


    global_file = (
        run_dir /
        "run.global.txt"
    )


    extra = [
        "",
        f"STARTYEAR   {current_day.year}",
        f"STARTMONTH  {current_day.month}",
        f"STARTDAY    {current_day.day}",
        "STARTSEC    0",

        "NRECS       24",

        f"RESULT_DIR  {result_dir}",

        "AGGFREQ     NHOURS 24",
    ]


    if init_state is not None:

        extra.append(
            f"INIT_STATE  {init_state}"
        )


    extra.extend([
        f"STATENAME   {state_dir / 'state'}",

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


    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

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
            str(
                KA_SCALE
            ),

        "VIC_GW_MAX_DRAIN_FRACTION":
            str(
                DRAIN_FRACTION
            ),

        "VIC_GW_EXCHANGE_LENGTH_M":
            str(
                EXCHANGE_LENGTH_M
            ),

        "VIC_GW_HEAD_FILE":
            str(
                head_file
            ),
    })


    env.pop(
        "VIC_GW_HEAD_OFFSET_M",
        None,
    )


    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

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
            f"VIC failed on day {day_index}; "
            f"see {run_dir / 'stderr.log'}"
        )


    fluxes = sorted(
        result_dir.glob(
            "fluxes*.nc"
        )
    )

    states = sorted(
        state_dir.glob(
            "state*.nc"
        )
    )


    if len(fluxes) != 1:

        raise RuntimeError(
            f"Expected one VIC flux file "
            f"for day {day_index}."
        )


    if len(states) != 1:

        raise RuntimeError(
            f"Expected one VIC state file "
            f"for day {day_index}."
        )


    # --------------------------------------------------------
    # Extract outputs
    # --------------------------------------------------------

    with nc.Dataset(
        fluxes[0]
    ) as ds:

        q_grid = np.ma.asarray(
            ds[
                "OUT_GW_EXCHANGE"
            ][0],
            dtype=float,
        )

        soil = np.ma.asarray(
            ds[
                "OUT_SOIL_MOIST"
            ][0],
            dtype=float,
        )

        water_error = np.ma.asarray(
            ds[
                "OUT_WATER_ERROR"
            ][0],
            dtype=float,
        )


    q_vic_mm = np.array(
        [
            float(
                q_grid[
                    vic_row[i],
                    vic_col[i]
                ]
            )
            for i in range(NV)
        ],
        dtype=float,
    )


    bottom = (
        soil[-1]
    )


    bottom_vic_mm = np.array(
        [
            float(
                bottom[
                    vic_row[i],
                    vic_col[i]
                ]
            )
            for i in range(NV)
        ],
        dtype=float,
    )


    wb_absmax = float(
        np.max(
            np.abs(
                water_error.compressed()
            )
        )
    )


    return {
        "q_vic_mm":
            q_vic_mm,

        "bottom_vic_mm":
            bottom_vic_mm,

        "water_error_absmax_mm":
            wb_absmax,

        "state_file":
            states[0],

        "flux_file":
            fluxes[0],

        "head_file":
            head_file,
    }


# ============================================================
# Write persistent 12-cell MF6 model
# ============================================================

area_text = " ".join(
    f"{x:.17g}"
    for x in mf6_area_m2
)


head_text = " ".join(
    f"{x:.17g}"
    for x in mf6_initial_head_m
)


iac_text = " ".join(
    "1"
    for _ in range(NM)
)


ja_text = " ".join(
    str(i)
    for i in range(
        1,
        NM + 1
    )
)


(MF6_DIR / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h7e.tdis
END TIMING

BEGIN MODELS
  GWF6 h7e.nam H7E
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h7e.ims H7E
END SOLUTIONGROUP
"""
)


(MF6_DIR / "h7e.tdis").write_text(
f"""BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  {NDAYS:.1f} {NDAYS} 1.0
END PERIODDATA
"""
)


(MF6_DIR / "h7e.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
)


(MF6_DIR / "h7e.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DISU6 h7e.disu DISU
  IC6   h7e.ic   IC
  NPF6  h7e.npf  NPF
  STO6  h7e.sto  STO
  API6  h7e.api  VICAPI
  OC6   h7e.oc   OC
END PACKAGES
"""
)


(MF6_DIR / "h7e.disu").write_text(
f"""BEGIN OPTIONS
  LENGTH_UNITS METERS
END OPTIONS

BEGIN DIMENSIONS
  NODES {NM}
  NJA {NM}
END DIMENSIONS

BEGIN GRIDDATA
  TOP
    CONSTANT {TOP_M:.17g}

  BOT
    CONSTANT {BOT_M:.17g}

  AREA
    INTERNAL FACTOR 1.0
    {area_text}
END GRIDDATA

BEGIN CONNECTIONDATA
  IAC
    INTERNAL FACTOR 1
    {iac_text}

  JA
    INTERNAL FACTOR 1
    {ja_text}

  IHC
    CONSTANT 0

  CL12
    CONSTANT 0.0

  HWVA
    CONSTANT 0.0
END CONNECTIONDATA
"""
)


(MF6_DIR / "h7e.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    INTERNAL FACTOR 1.0
    {head_text}
END GRIDDATA
"""
)


(MF6_DIR / "h7e.npf").write_text(
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


(MF6_DIR / "h7e.sto").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICONVERT
    CONSTANT 0

  SS
    CONSTANT {SS_PER_M:.17g}
END GRIDDATA

BEGIN PERIOD 1
  TRANSIENT
END PERIOD
"""
)


(MF6_DIR / "h7e.api").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND {NM}
END DIMENSIONS
"""
)


(MF6_DIR / "h7e.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h7e.hds
  BUDGET FILEOUT h7e.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
)


# ============================================================
# Persistent coupled run
# ============================================================

mf6 = XmiWrapper(
    str(LIBMF6),
    working_directory=
        str(MF6_DIR),
)


mf6.initialize_mpi(MPI.COMM_SELF.py2f())


daily_rows = []
cell_rows = []

accepted_vic_state = None

cumulative_vic_m3 = 0.0

max_daily_cross_error = 0.0
max_cell_storage_error = 0.0
max_api_error = 0.0

day1_regression_pass = None


try:

    head = mf6.get_value_ptr(
        mf6.get_var_address(
            "X",
            "H7E"
        )
    )


    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H7E",
            "VICAPI"
        )
    )


    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H7E",
            "VICAPI"
        )
    )


    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H7E",
            "VICAPI"
        )
    )


    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H7E",
            "VICAPI"
        )
    )


    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H7E",
            "VICAPI"
        )
    )


    actual_initial_head = np.array(
        head,
        dtype=float,
        copy=True,
    )


    if not np.allclose(
        actual_initial_head,
        mf6_initial_head_m,
        rtol=0.0,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "MF6 initial heads do not match "
            "the H7b heterogeneous field."
        )


    for day_index in range(
        1,
        NDAYS + 1
    ):

        current_day = (
            START +
            timedelta(
                days=
                    day_index - 1
            )
        )


        mf6_head_start = np.array(
            head,
            dtype=float,
            copy=True,
        )


        # ----------------------------------------------------
        # MF6 -> VIC
        # ----------------------------------------------------

        vic_head_start = (
            A.dot(
                mf6_head_start
            ) /
            vic_area_m2
        )


        print()
        print("=" * 84)

        print(
            f"H7e DAY {day_index:02d} "
            f"{current_day}"
        )

        print("=" * 84)

        print(
            "MF6 head range =",
            mf6_head_start.min(),
            mf6_head_start.max(),
            "m"
        )

        print(
            "VIC head range =",
            vic_head_start.min(),
            vic_head_start.max(),
            "m"
        )


        # ----------------------------------------------------
        # Run VIC from accepted previous-day state
        # ----------------------------------------------------

        vic_result = run_vic_day(
            day_index,
            current_day,
            vic_head_start,
            accepted_vic_state,
        )


        q_vic_mm = (
            vic_result[
                "q_vic_mm"
            ]
        )


        vic_cell_volume_m3 = (
            q_vic_mm *
            1.0e-3 *
            vic_area_m2
        )


        vic_total_m3 = float(
            vic_cell_volume_m3.sum()
        )


        vic_positive_m3 = float(
            np.sum(
                np.maximum(
                    q_vic_mm,
                    0.0
                ) *
                1.0e-3 *
                vic_area_m2
            )
        )


        vic_negative_m3 = float(
            np.sum(
                np.minimum(
                    q_vic_mm,
                    0.0
                ) *
                1.0e-3 *
                vic_area_m2
            )
        )


        # ----------------------------------------------------
        # VIC -> MF6 overlap mapping
        # ----------------------------------------------------

        mf6_volume_m3 = (
            1.0e-3 *
            A.T.dot(
                q_vic_mm
            )
        )


        mf6_positive_m3 = (
            1.0e-3 *
            A.T.dot(
                np.maximum(
                    q_vic_mm,
                    0.0
                )
            )
        )


        mf6_negative_m3 = (
            1.0e-3 *
            A.T.dot(
                np.minimum(
                    q_vic_mm,
                    0.0
                )
            )
        )


        mapping_error = (
            float(
                mf6_volume_m3.sum()
            ) -
            vic_total_m3
        )


        positive_mapping_error = (
            float(
                mf6_positive_m3.sum()
            ) -
            vic_positive_m3
        )


        negative_mapping_error = (
            float(
                mf6_negative_m3.sum()
            ) -
            vic_negative_m3
        )


        if abs(
            mapping_error
        ) > 1.0e-6:

            raise RuntimeError(
                f"Net spatial mapping failed "
                f"on day {day_index}."
            )


        if (
            abs(
                positive_mapping_error
            ) > 1.0e-6 or
            abs(
                negative_mapping_error
            ) > 1.0e-6
        ):

            raise RuntimeError(
                f"Gross signed mapping failed "
                f"on day {day_index}."
            )


        # ----------------------------------------------------
        # MF6 time step
        # ----------------------------------------------------

        expected_head_after = (
            mf6_head_start +
            mf6_volume_m3 /
            storage_coeff_m2
        )


        mf6.prepare_time_step(
            DT_DAYS
        )


        nbound[0] = NM

        nodelist[:NM] = np.arange(
            1,
            NM + 1,
            dtype=nodelist.dtype,
        )

        hcof[:NM] = 0.0

        rhs[:NM] = (
            -mf6_volume_m3 /
            DT_DAYS
        )


        component = 1

        mf6.prepare_solve(
            component
        )


        converged = False
        mf6_iterations = 0


        for iteration in range(
            1,
            101
        ):

            nbound[0] = NM

            nodelist[:NM] = np.arange(
                1,
                NM + 1,
                dtype=nodelist.dtype,
            )

            hcof[:NM] = 0.0

            rhs[:NM] = (
                -mf6_volume_m3 /
                DT_DAYS
            )


            converged = mf6.solve(
                component
            )

            mf6_iterations = (
                iteration
            )


            if converged:
                break


        if not converged:

            raise RuntimeError(
                f"MF6 failed on day {day_index}."
            )


        mf6.finalize_solve(
            component
        )

        mf6.finalize_time_step()


        mf6_head_after = np.array(
            head,
            dtype=float,
            copy=True,
        )


        api_flow = np.array(
            simvals[:NM],
            dtype=float,
            copy=True,
        )


        # ----------------------------------------------------
        # Verify MF6 response
        # ----------------------------------------------------

        actual_dh = (
            mf6_head_after -
            mf6_head_start
        )


        storage_change = (
            actual_dh *
            storage_coeff_m2
        )


        cell_storage_error = (
            storage_change -
            mf6_volume_m3
        )


        api_error = (
            api_flow -
            mf6_volume_m3 /
            DT_DAYS
        )


        head_prediction_error = (
            mf6_head_after -
            expected_head_after
        )


        daily_storage_total = float(
            storage_change.sum()
        )


        daily_cross_error = (
            daily_storage_total -
            vic_total_m3
        )


        relative_daily_error = (
            daily_cross_error /
            vic_total_m3
            if vic_total_m3 != 0.0
            else 0.0
        )


        max_day_cell_error = float(
            np.max(
                np.abs(
                    cell_storage_error
                )
            )
        )


        max_day_api_error = float(
            np.max(
                np.abs(
                    api_error
                )
            )
        )


        max_day_head_error = float(
            np.max(
                np.abs(
                    head_prediction_error
                )
            )
        )


        max_daily_cross_error = max(
            max_daily_cross_error,
            abs(
                daily_cross_error
            )
        )


        max_cell_storage_error = max(
            max_cell_storage_error,
            max_day_cell_error,
        )


        max_api_error = max(
            max_api_error,
            max_day_api_error,
        )


        # ----------------------------------------------------
        # Updated MF6 -> VIC map
        # ----------------------------------------------------

        vic_head_after = (
            A.dot(
                mf6_head_after
            ) /
            vic_area_m2
        )


        vic_head_change = (
            vic_head_after -
            vic_head_start
        )


        # ----------------------------------------------------
        # H7d day-1 regression
        # ----------------------------------------------------

        if (
            day_index == 1 and
            H7D_REF is not None
        ):

            q_diff = float(
                np.max(
                    np.abs(
                        q_vic_mm -
                        h7d_vic_q
                    )
                )
            )

            volume_diff = float(
                np.max(
                    np.abs(
                        mf6_volume_m3 -
                        h7d_mf6_volume
                    )
                )
            )

            mf6_head_diff = float(
                np.max(
                    np.abs(
                        mf6_head_after -
                        h7d_mf6_head
                    )
                )
            )

            vic_head_diff = float(
                np.max(
                    np.abs(
                        vic_head_after -
                        h7d_vic_return_head
                    )
                )
            )


            day1_regression_pass = (
                q_diff == 0.0 and
                volume_diff < 1.0e-6 and
                mf6_head_diff < 1.0e-12 and
                vic_head_diff < 1.0e-12
            )


            print()
            print(
                "H7d day-1 regression:"
            )

            print(
                "  max q diff        =",
                q_diff,
                "mm"
            )

            print(
                "  max mapped V diff =",
                volume_diff,
                "m3"
            )

            print(
                "  max MF6 head diff =",
                mf6_head_diff,
                "m"
            )

            print(
                "  max VIC head diff =",
                vic_head_diff,
                "m"
            )

            print(
                "  result            =",
                "PASS"
                if day1_regression_pass
                else "FAIL"
            )


        # ----------------------------------------------------
        # Save daily summary
        # ----------------------------------------------------

        cumulative_vic_m3 += (
            vic_total_m3
        )


        daily_rows.append({
            "day":
                day_index,

            "date":
                current_day.isoformat(),

            "mf6_iterations":
                mf6_iterations,

            "mf6_head_start_min_m":
                float(
                    mf6_head_start.min()
                ),

            "mf6_head_start_max_m":
                float(
                    mf6_head_start.max()
                ),

            "vic_head_start_min_m":
                float(
                    vic_head_start.min()
                ),

            "vic_head_start_max_m":
                float(
                    vic_head_start.max()
                ),

            "vic_q_min_mm":
                float(
                    q_vic_mm.min()
                ),

            "vic_q_max_mm":
                float(
                    q_vic_mm.max()
                ),

            "vic_positive_cells":
                int(
                    np.sum(
                        q_vic_mm > 0.0
                    )
                ),

            "vic_negative_cells":
                int(
                    np.sum(
                        q_vic_mm < 0.0
                    )
                ),

            "vic_positive_volume_m3":
                vic_positive_m3,

            "vic_negative_volume_m3":
                vic_negative_m3,

            "vic_net_volume_m3":
                vic_total_m3,

            "mapped_mf6_volume_m3":
                float(
                    mf6_volume_m3.sum()
                ),

            "mapping_error_m3":
                mapping_error,

            "positive_mapping_error_m3":
                positive_mapping_error,

            "negative_mapping_error_m3":
                negative_mapping_error,

            "mf6_storage_change_m3":
                daily_storage_total,

            "cross_model_error_m3":
                daily_cross_error,

            "relative_cross_model_error":
                relative_daily_error,

            "max_cell_storage_error_m3":
                max_day_cell_error,

            "max_api_error_m3_day":
                max_day_api_error,

            "max_head_prediction_error_m":
                max_day_head_error,

            "mf6_head_end_min_m":
                float(
                    mf6_head_after.min()
                ),

            "mf6_head_end_max_m":
                float(
                    mf6_head_after.max()
                ),

            "vic_head_end_min_m":
                float(
                    vic_head_after.min()
                ),

            "vic_head_end_max_m":
                float(
                    vic_head_after.max()
                ),

            "vic_head_change_min_m":
                float(
                    vic_head_change.min()
                ),

            "vic_head_change_max_m":
                float(
                    vic_head_change.max()
                ),

            "bottom_moist_min_mm":
                float(
                    vic_result[
                        "bottom_vic_mm"
                    ].min()
                ),

            "bottom_moist_max_mm":
                float(
                    vic_result[
                        "bottom_vic_mm"
                    ].max()
                ),

            "vic_water_error_absmax_mm":
                vic_result[
                    "water_error_absmax_mm"
                ],

            "cumulative_vic_volume_m3":
                cumulative_vic_m3,

            "vic_state_file":
                str(
                    vic_result[
                        "state_file"
                    ]
                ),
        })


        # ----------------------------------------------------
        # Save every MF6-cell/day response
        # ----------------------------------------------------

        for j in range(NM):

            cell_rows.append({
                "day":
                    day_index,

                "date":
                    current_day.isoformat(),

                "mf6_id":
                    j,

                "full_area_m2":
                    mf6_area_m2[j],

                "coupled_area_m2":
                    mf6_coupled_area_m2[j],

                "head_start_m":
                    mf6_head_start[j],

                "mapped_positive_m3":
                    mf6_positive_m3[j],

                "mapped_negative_m3":
                    mf6_negative_m3[j],

                "mapped_net_m3":
                    mf6_volume_m3[j],

                "api_simvals_m3_day":
                    api_flow[j],

                "head_end_m":
                    mf6_head_after[j],

                "storage_change_m3":
                    storage_change[j],

                "storage_error_m3":
                    cell_storage_error[j],
            })


        # ----------------------------------------------------
        # Advance accepted VIC state
        # ----------------------------------------------------

        accepted_vic_state = (
            vic_result[
                "state_file"
            ]
        )


        print()
        print(
            "VIC q range       =",
            q_vic_mm.min(),
            q_vic_mm.max(),
            "mm/day"
        )

        print(
            "positive/negative =",
            int(
                np.sum(
                    q_vic_mm > 0.0
                )
            ),
            "/",
            int(
                np.sum(
                    q_vic_mm < 0.0
                )
            )
        )

        print(
            "VIC transfer      =",
            vic_total_m3,
            "m3"
        )

        print(
            "MF6 storage       =",
            daily_storage_total,
            "m3"
        )

        print(
            "cross-model error =",
            daily_cross_error,
            "m3"
        )

        print(
            "MF6 end heads     =",
            mf6_head_after.min(),
            mf6_head_after.max(),
            "m"
        )

        print(
            "VIC return heads  =",
            vic_head_after.min(),
            vic_head_after.max(),
            "m"
        )


    final_mf6_head_m = np.array(
        head,
        dtype=float,
        copy=True,
    )


finally:

    mf6.finalize()


# ============================================================
# Cumulative conservation
# ============================================================

cumulative_mf6_storage_m3 = float(
    np.sum(
        (
            final_mf6_head_m -
            actual_initial_head
        ) *
        storage_coeff_m2
    )
)


cumulative_cross_error_m3 = (
    cumulative_mf6_storage_m3 -
    cumulative_vic_m3
)


relative_cumulative_error = (
    cumulative_cross_error_m3 /
    cumulative_vic_m3
)


final_vic_head_m = (
    A.dot(
        final_mf6_head_m
    ) /
    vic_area_m2
)


# ============================================================
# Save final head field
# ============================================================

FINAL_HEAD_FILE = (
    OUT /
    "vic_head_final.txt"
)


with FINAL_HEAD_FILE.open(
    "w"
) as f:

    f.write(
        "# latitude longitude head_offset_m\n"
    )

    for i in range(NV):

        f.write(
            f"{vic_lat[i]:.17g} "
            f"{vic_lon[i]:.17g} "
            f"{final_vic_head_m[i]:.17g}\n"
        )


# ============================================================
# Save CSVs
# ============================================================

with (
    OUT /
    "daily.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            daily_rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        daily_rows
    )


with (
    OUT /
    "mf6_cell_daily.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            cell_rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        cell_rows
    )


# ============================================================
# Final pass/fail
# ============================================================

DAILY_CONSERVATION_PASS = (
    max_daily_cross_error <
    1.0e-3
)


CELL_STORAGE_PASS = (
    max_cell_storage_error <
    1.0e-3
)


API_PASS = (
    max_api_error <
    1.0e-6
)


CUMULATIVE_PASS = (
    abs(
        cumulative_cross_error_m3
    ) <
    1.0e-3
)


WATER_BALANCE_PASS = (
    max(
        r[
            "vic_water_error_absmax_mm"
        ]
        for r in daily_rows
    ) <
    1.0e-8
)


REGRESSION_PASS = (
    True
    if day1_regression_pass is None
    else day1_regression_pass
)


OVERALL = all([
    DAILY_CONSERVATION_PASS,
    CELL_STORAGE_PASS,
    API_PASS,
    CUMULATIVE_PASS,
    WATER_BALANCE_PASS,
    REGRESSION_PASS,
])


summary = {
    "days":
        NDAYS,

    "active_vic_cells":
        NV,

    "mf6_cells":
        NM,

    "overlap_records":
        int(
            np.count_nonzero(A)
        ),

    "Ss_per_m":
        SS_PER_M,

    "initial_mf6_head_min_m":
        float(
            actual_initial_head.min()
        ),

    "initial_mf6_head_max_m":
        float(
            actual_initial_head.max()
        ),

    "final_mf6_head_min_m":
        float(
            final_mf6_head_m.min()
        ),

    "final_mf6_head_max_m":
        float(
            final_mf6_head_m.max()
        ),

    "final_vic_head_min_m":
        float(
            final_vic_head_m.min()
        ),

    "final_vic_head_max_m":
        float(
            final_vic_head_m.max()
        ),

    "cumulative_vic_exchange_m3":
        cumulative_vic_m3,

    "cumulative_mf6_storage_change_m3":
        cumulative_mf6_storage_m3,

    "cumulative_cross_model_error_m3":
        cumulative_cross_error_m3,

    "relative_cumulative_error":
        relative_cumulative_error,

    "max_daily_cross_model_error_m3":
        max_daily_cross_error,

    "max_cell_storage_error_m3":
        max_cell_storage_error,

    "max_api_error_m3_day":
        max_api_error,

    "max_vic_water_error_mm":
        max(
            r[
                "vic_water_error_absmax_mm"
            ]
            for r in daily_rows
        ),

    "day1_h7d_regression_pass":
        day1_regression_pass,

    "overall_pass":
        OVERALL,
}


with (
    OUT /
    "summary.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            summary.keys()
    )

    writer.writeheader()

    writer.writerow(
        summary
    )


# ============================================================
# Print final table
# ============================================================

print()
print("=" * 132)
print("H7e 10-DAY SPATIAL CLOSED LOOP")
print("=" * 132)

print(
    f"{'day':>3s} "
    f"{'q min':>10s} "
    f"{'q max':>10s} "
    f"{'+/-':>7s} "
    f"{'VIC volume':>15s} "
    f"{'MF6 storage':>15s} "
    f"{'error':>12s} "
    f"{'MF6 h min':>12s} "
    f"{'MF6 h max':>12s}"
)


for r in daily_rows:

    print(
        f"{r['day']:3d} "
        f"{r['vic_q_min_mm']:10.4f} "
        f"{r['vic_q_max_mm']:10.4f} "
        f"{r['vic_positive_cells']:2d}/"
        f"{r['vic_negative_cells']:<2d} "
        f"{r['vic_net_volume_m3']:15.3f} "
        f"{r['mf6_storage_change_m3']:15.3f} "
        f"{r['cross_model_error_m3']:12.3e} "
        f"{r['mf6_head_end_min_m']:12.5f} "
        f"{r['mf6_head_end_max_m']:12.5f}"
    )


print()
print("=" * 84)
print("H7e VALIDATION")
print("=" * 84)

print(
    "H7d day-1 regression:",
    (
        "PASS"
        if REGRESSION_PASS
        else "FAIL"
    )
)

print(
    "daily conservation:",
    (
        "PASS"
        if DAILY_CONSERVATION_PASS
        else "FAIL"
    ),
    "max error =",
    max_daily_cross_error,
    "m3"
)

print(
    "per-cell storage:",
    (
        "PASS"
        if CELL_STORAGE_PASS
        else "FAIL"
    ),
    "max error =",
    max_cell_storage_error,
    "m3"
)

print(
    "API flows:",
    (
        "PASS"
        if API_PASS
        else "FAIL"
    ),
    "max error =",
    max_api_error,
    "m3/day"
)

print(
    "VIC water balance:",
    (
        "PASS"
        if WATER_BALANCE_PASS
        else "FAIL"
    )
)

print(
    "cumulative conservation:",
    (
        "PASS"
        if CUMULATIVE_PASS
        else "FAIL"
    )
)


print()
print(
    "cumulative VIC exchange =",
    cumulative_vic_m3,
    "m3"
)

print(
    "cumulative MF6 storage  =",
    cumulative_mf6_storage_m3,
    "m3"
)

print(
    "cumulative error        =",
    cumulative_cross_error_m3,
    "m3"
)

print(
    "relative error          =",
    relative_cumulative_error
)


print()
print(
    "initial MF6 head range =",
    actual_initial_head.min(),
    actual_initial_head.max(),
    "m"
)

print(
    "final MF6 head range   =",
    final_mf6_head_m.min(),
    final_mf6_head_m.max(),
    "m"
)

print(
    "final VIC head range   =",
    final_vic_head_m.min(),
    final_vic_head_m.max(),
    "m"
)


print()
print("=" * 84)

print(
    "H7e MULTIDAY SPATIAL TWO-WAY COUPLING:",
    (
        "PASS"
        if OVERALL
        else "REQUIRES REVIEW"
    )
)

print("=" * 84)


print()
print("Saved:")

print(
    OUT /
    "daily.csv"
)

print(
    OUT /
    "mf6_cell_daily.csv"
)

print(
    OUT /
    "summary.csv"
)

print(
    FINAL_HEAD_FILE
)

print()
print(
    "Final VIC restart =",
    accepted_vic_state
)

print()
print(
    "Experiment directory:"
)

print(
    OUT
)
