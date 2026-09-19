#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
from datetime import date, timedelta
import csv
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
    f"H8c_connected_10day_closed_loop_{STAMP}"
)

MF6_DIR = (
    OUT /
    "mf6"
)

VIC_OUT = (
    OUT /
    "vic"
)

MF6_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIC_OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experiment controls
# ============================================================

START_DATE = date(
    1949,
    1,
    1,
)

NDAYS = 10

NROW = 3
NCOL = 4
NODES = NROW * NCOL

DT_DAYS = 1.0

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)

SS_PER_M = 1.0e-5

KA_SCALE = 0.001

EXCHANGE_LENGTH_M = 100.0

DRAIN_FRACTION = 1.0


# ============================================================
# Locate validated H8a connected model
# ============================================================

h8a_candidates = sorted(
    ROOT.glob(
        "H8a_connected_mf6_lateral_*"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)


if not h8a_candidates:

    raise RuntimeError(
        "No H8a connected MF6 experiment found."
    )


H8A = h8a_candidates[0]


print()
print(
    "H8a connected model =",
    H8A
)


# ============================================================
# Locate most recent PASSING H8b for day-1 regression
# ============================================================

H8B = None


for candidate in sorted(
    ROOT.glob(
        "H8b_connected_interface_budget_*"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
):

    summary_file = (
        candidate /
        "summary.csv"
    )

    if not summary_file.exists():
        continue


    with summary_file.open() as f:

        rows = list(
            csv.DictReader(f)
        )


    if not rows:
        continue


    value = str(
        rows[0].get(
            "overall_pass",
            ""
        )
    ).strip().lower()


    if value in (
        "true",
        "1",
        "yes",
    ):

        H8B = candidate
        break


if H8B is None:

    raise RuntimeError(
        "Could not find a passing H8b run."
    )


print(
    "H8b regression ref =",
    H8B
)


# ============================================================
# Read actual VIC-cell geometry
# ============================================================

with (
    H7B /
    "vic_cells.csv"
).open() as f:

    vic_rows = list(
        csv.DictReader(f)
    )


vic_rows.sort(
    key=lambda x:
        int(x["vic_id"])
)


if len(vic_rows) != 16:

    raise RuntimeError(
        f"Expected 16 VIC cells; "
        f"found {len(vic_rows)}."
    )


NV = len(
    vic_rows
)


vic_row = np.array(
    [
        int(x["row"])
        for x in vic_rows
    ],
    dtype=int,
)


vic_col = np.array(
    [
        int(x["col"])
        for x in vic_rows
    ],
    dtype=int,
)


vic_lat = np.array(
    [
        float(x["lat"])
        for x in vic_rows
    ],
    dtype=float,
)


vic_lon = np.array(
    [
        float(x["lon"])
        for x in vic_rows
    ],
    dtype=float,
)


vic_area_m2 = np.array(
    [
        float(x["area_m2"])
        for x in vic_rows
    ],
    dtype=float,
)


# ============================================================
# Read H7b MF6-cell geometry / initial head field
# ============================================================

with (
    H7B /
    "mf6_cells.csv"
).open() as f:

    mf6_rows = list(
        csv.DictReader(f)
    )


mf6_rows.sort(
    key=lambda x:
        int(x["mf6_id"])
)


if len(mf6_rows) != NODES:

    raise RuntimeError(
        f"Expected {NODES} MF6 cells; "
        f"found {len(mf6_rows)}."
    )


mf6_area_m2 = np.array(
    [
        float(
            x["rectangle_area_m2"]
        )
        for x in mf6_rows
    ],
    dtype=float,
)


mf6_coupled_area_m2 = np.array(
    [
        float(
            x["coupled_area_m2"]
        )
        for x in mf6_rows
    ],
    dtype=float,
)


mf6_initial_head_m = np.array(
    [
        float(
            x["head_m"]
        )
        for x in mf6_rows
    ],
    dtype=float,
)


# ============================================================
# H7b overlap matrix
# ============================================================

A = np.zeros(
    (
        NV,
        NODES
    ),
    dtype=float,
)


with (
    H7B /
    "exchange_table.csv"
).open() as f:

    overlap_rows = list(
        csv.DictReader(f)
    )


for x in overlap_rows:

    i = int(
        x["vic_id"]
    )

    j = int(
        x["mf6_id"]
    )

    A[i, j] = float(
        x["overlap_area_m2"]
    )


if not np.allclose(
    A.sum(axis=1),
    vic_area_m2,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "H7b overlap coverage failed."
    )


# ============================================================
# Reconstruct validated H8 topology for FLOWJA decoding
# ============================================================

def node(
    row,
    col,
):

    return (
        row *
        NCOL +
        col
    )


neighbors = {
    n: []
    for n in range(
        NODES
    )
}


for r in range(
    NROW
):

    for c in range(
        NCOL
    ):

        n = node(
            r,
            c
        )


        if c > 0:

            neighbors[n].append(
                node(
                    r,
                    c - 1
                )
            )


        if c < NCOL - 1:

            neighbors[n].append(
                node(
                    r,
                    c + 1
                )
            )


        if r > 0:

            neighbors[n].append(
                node(
                    r - 1,
                    c
                )
            )


        if r < NROW - 1:

            neighbors[n].append(
                node(
                    r + 1,
                    c
                )
            )


        neighbors[n].sort()


iac = np.array(
    [
        1 +
        len(
            neighbors[n]
        )
        for n in range(
            NODES
        )
    ],
    dtype=int,
)


ja = []


for n in range(
    NODES
):

    # Self entry first.
    ja.append(
        n
    )

    ja.extend(
        neighbors[n]
    )


ja = np.asarray(
    ja,
    dtype=int,
)


NJA = int(
    iac.sum()
)


if NJA != 46:

    raise RuntimeError(
        f"Expected NJA=46; got {NJA}."
    )


offset = np.zeros(
    NODES,
    dtype=int,
)


for n in range(
    1,
    NODES
):

    offset[n] = (
        offset[n - 1] +
        iac[n - 1]
    )


# ============================================================
# Storage
# ============================================================

storage_coeff_m2 = (
    SS_PER_M *
    THICKNESS_M *
    mf6_area_m2
)


# ============================================================
# Read H8b day-1 regression values
# ============================================================

with (
    H8B /
    "cell_budget.csv"
).open() as f:

    h8b_rows = list(
        csv.DictReader(f)
    )


h8b_rows.sort(
    key=lambda x:
        int(x["mf6_id"])
)


h8b_interface_m3 = np.array(
    [
        float(
            x["interface_volume_m3"]
        )
        for x in h8b_rows
    ],
    dtype=float,
)


h8b_lateral_m3 = np.array(
    [
        float(
            x["lateral_net_volume_m3"]
        )
        for x in h8b_rows
    ],
    dtype=float,
)


h8b_head_after_m = np.array(
    [
        float(
            x["head_after_connected_m"]
        )
        for x in h8b_rows
    ],
    dtype=float,
)


h8b_storage_m3 = np.array(
    [
        float(
            x["storage_change_m3"]
        )
        for x in h8b_rows
    ],
    dtype=float,
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

    "AGGFREQ",
}


base_lines = []


for line in (
    BASE_GLOBAL
    .read_text()
    .splitlines()
):

    stripped = (
        line.strip()
    )


    if (
        stripped and
        not stripped.startswith("#")
    ):

        if (
            stripped
            .split()[0]
            in controlled
        ):

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
# One-day VIC runner
# ============================================================

def run_vic_day(
    day_index,
    current_date,
    vic_head_m,
    init_state,
):

    run_dir = (
        VIC_OUT /
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
    # Spatial MF6 -> VIC head file
    # --------------------------------------------------------

    head_file = (
        run_dir /
        "vic_gw_head.txt"
    )


    with head_file.open(
        "w"
    ) as f:

        f.write(
            "# latitude longitude "
            "head_offset_m\n"
        )


        for i in range(
            NV
        ):

            f.write(
                f"{vic_lat[i]:.17g} "
                f"{vic_lon[i]:.17g} "
                f"{vic_head_m[i]:.17g}\n"
            )


    # --------------------------------------------------------
    # VIC global file
    # --------------------------------------------------------

    next_date = (
        current_date +
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
        f"STARTYEAR   {current_date.year}",
        f"STARTMONTH  {current_date.month}",
        f"STARTDAY    {current_date.day}",
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

        f"STATEYEAR   {next_date.year}",
        f"STATEMONTH  {next_date.month}",
        f"STATEDAY    {next_date.day}",
        "STATESEC    0",

        "STATE_FORMAT NETCDF4_CLASSIC",
        "",
    ])


    global_file.write_text(
        BASE_TEXT +
        "\n".join(
            extra
        )
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
    # Execute
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
                str(
                    VIC_EXE
                ),
                "-g",
                str(
                    global_file
                ),
            ],
            cwd=
                PARAMDIR,
            env=
                env,
            stdout=
                stdout,
            stderr=
                stderr,
        )


    if proc.returncode != 0:

        raise RuntimeError(
            f"VIC failed on day "
            f"{day_index}; see "
            f"{run_dir / 'stderr.log'}"
        )


    flux_files = sorted(
        result_dir.glob(
            "fluxes*.nc"
        )
    )


    state_files = sorted(
        state_dir.glob(
            "state*.nc"
        )
    )


    if len(
        flux_files
    ) != 1:

        raise RuntimeError(
            f"Expected one flux file "
            f"for day {day_index}."
        )


    if len(
        state_files
    ) != 1:

        raise RuntimeError(
            f"Expected one state file "
            f"for day {day_index}."
        )


    # --------------------------------------------------------
    # Read VIC outputs
    # --------------------------------------------------------

    with nc.Dataset(
        flux_files[0]
    ) as ds:

        q_grid = np.ma.asarray(
            ds[
                "OUT_GW_EXCHANGE"
            ][0],
            dtype=float,
        )


        water_error = np.ma.asarray(
            ds[
                "OUT_WATER_ERROR"
            ][0],
            dtype=float,
        )


        soil = np.ma.asarray(
            ds[
                "OUT_SOIL_MOIST"
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
            for i in range(
                NV
            )
        ],
        dtype=float,
    )


    bottom_grid = (
        soil[-1]
    )


    bottom_moist_mm = np.array(
        [
            float(
                bottom_grid[
                    vic_row[i],
                    vic_col[i]
                ]
            )
            for i in range(
                NV
            )
        ],
        dtype=float,
    )


    wb_absmax = float(
        np.max(
            np.abs(
                water_error
                .compressed()
            )
        )
    )


    return {
        "q_vic_mm":
            q_vic_mm,

        "bottom_moist_mm":
            bottom_moist_mm,

        "water_error_absmax_mm":
            wb_absmax,

        "state_file":
            state_files[0],

        "flux_file":
            flux_files[0],

        "head_file":
            head_file,
    }


# ============================================================
# Copy exact validated H8a connected groundwater packages
# ============================================================

H8A_MF6 = (
    H8A /
    "mf6"
)


for src, dst in [
    (
        "h8a.disu",
        "h8c.disu",
    ),
    (
        "h8a.ic",
        "h8c.ic",
    ),
    (
        "h8a.npf",
        "h8c.npf",
    ),
    (
        "h8a.sto",
        "h8c.sto",
    ),
    (
        "h8a.ims",
        "h8c.ims",
    ),
]:

    shutil.copyfile(
        H8A_MF6 /
        src,

        MF6_DIR /
        dst,
    )


# ============================================================
# Persistent 10-day MF6 simulation
# ============================================================

(MF6_DIR / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h8c.tdis
END TIMING

BEGIN MODELS
  GWF6 h8c.nam H8C
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h8c.ims H8C
END SOLUTIONGROUP
"""
)


(MF6_DIR / "h8c.tdis").write_text(
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


(MF6_DIR / "h8c.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DISU6 h8c.disu DISU
  IC6   h8c.ic   IC
  NPF6  h8c.npf  NPF
  STO6  h8c.sto  STO
  API6  h8c.api  VICAPI
  OC6   h8c.oc   OC
END PACKAGES
"""
)


(MF6_DIR / "h8c.api").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND {NODES}
END DIMENSIONS
"""
)


(MF6_DIR / "h8c.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h8c.hds
  BUDGET FILEOUT h8c.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
)


# ============================================================
# FLOWJA decoding helper
# ============================================================

def decode_lateral_flow(
    flowja_values,
):

    lateral_rate = np.zeros(
        NODES,
        dtype=float,
    )


    directed = {}


    for n in range(
        NODES
    ):

        start = (
            offset[n]
        )

        end = (
            start +
            iac[n]
        )


        if (
            ja[start] !=
            n
        ):

            raise RuntimeError(
                f"JA row {n} does not "
                "begin with self."
            )


        # Skip diagonal/self entry.
        for pos in range(
            start + 1,
            end
        ):

            m = int(
                ja[pos]
            )


            q = float(
                flowja_values[
                    pos
                ]
            )


            directed[
                (
                    n,
                    m
                )
            ] = q


            # H8b established this sign:
            #
            # positive FLOWJA =
            # flow INTO current cell n.
            lateral_rate[n] += (
                q
            )


    pair_errors = []


    seen = set()


    for (
        n,
        m
    ), q_nm in directed.items():

        pair = tuple(
            sorted(
                (
                    n,
                    m
                )
            )
        )


        if pair in seen:

            continue


        seen.add(
            pair
        )


        q_mn = directed[
            (
                m,
                n
            )
        ]


        pair_errors.append(
            q_nm +
            q_mn
        )


    max_pair_error = float(
        np.max(
            np.abs(
                pair_errors
            )
        )
    )


    return (
        lateral_rate,
        max_pair_error,
    )


# ============================================================
# Persistent coupled simulation
# ============================================================

mf6 = XmiWrapper(
    str(
        LIBMF6
    ),
    working_directory=
        str(
            MF6_DIR
        ),
)


mf6.initialize_mpi(MPI.COMM_SELF.py2f())


daily_rows = []
cell_rows = []

accepted_vic_state = None

cumulative_vic_m3 = 0.0

max_cell_budget_error = 0.0
max_domain_error = 0.0
max_api_error = 0.0
max_pair_error = 0.0
max_net_lateral = 0.0
max_water_error = 0.0

day1_regression_pass = False


try:

    # --------------------------------------------------------
    # XMI pointers
    # --------------------------------------------------------

    head = mf6.get_value_ptr(
        mf6.get_var_address(
            "X",
            "H8C"
        )
    )


    flowja = mf6.get_value_ptr(
        mf6.get_var_address(
            "FLOWJA",
            "H8C"
        )
    )


    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H8C",
            "VICAPI"
        )
    )


    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H8C",
            "VICAPI"
        )
    )


    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H8C",
            "VICAPI"
        )
    )


    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H8C",
            "VICAPI"
        )
    )


    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H8C",
            "VICAPI"
        )
    )


    if (
        flowja.size !=
        NJA
    ):

        raise RuntimeError(
            f"FLOWJA={flowja.size}; "
            f"NJA={NJA}."
        )


    initial_head_actual = np.array(
        head,
        dtype=float,
        copy=True,
    )


    if not np.allclose(
        initial_head_actual,
        mf6_initial_head_m,
        rtol=0.0,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "H8c MF6 initial-head mismatch."
        )


    # --------------------------------------------------------
    # Ten accepted coupling steps
    # --------------------------------------------------------

    for day_index in range(
        1,
        NDAYS + 1
    ):

        current_date = (
            START_DATE +
            timedelta(
                days=
                    day_index - 1
            )
        )


        print()
        print("=" * 92)

        print(
            f"H8c DAY {day_index:02d} "
            f"{current_date}"
        )

        print("=" * 92)


        mf6_head_start = np.array(
            head,
            dtype=float,
            copy=True,
        )


        # ----------------------------------------------------
        # MF6 -> VIC head mapping
        # ----------------------------------------------------

        vic_head_start = (
            A.dot(
                mf6_head_start
            ) /
            vic_area_m2
        )


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
        # VIC one-day accepted trajectory
        # ----------------------------------------------------

        vic_result = run_vic_day(
            day_index,
            current_date,
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
        # VIC -> MF6 conservative overlap mapping
        # ----------------------------------------------------

        interface_m3 = (
            1.0e-3 *
            A.T.dot(
                q_vic_mm
            )
        )


        interface_positive_m3 = (
            1.0e-3 *
            A.T.dot(
                np.maximum(
                    q_vic_mm,
                    0.0
                )
            )
        )


        interface_negative_m3 = (
            1.0e-3 *
            A.T.dot(
                np.minimum(
                    q_vic_mm,
                    0.0
                )
            )
        )


        mapped_total_m3 = float(
            interface_m3.sum()
        )


        mapping_error_m3 = (
            mapped_total_m3 -
            vic_total_m3
        )


        positive_mapping_error_m3 = (
            float(
                interface_positive_m3.sum()
            ) -
            vic_positive_m3
        )


        negative_mapping_error_m3 = (
            float(
                interface_negative_m3.sum()
            ) -
            vic_negative_m3
        )


        if abs(
            mapping_error_m3
        ) > 1.0e-6:

            raise RuntimeError(
                f"Day {day_index}: "
                "net overlap mapping failed."
            )


        if (
            abs(
                positive_mapping_error_m3
            ) > 1.0e-6 or
            abs(
                negative_mapping_error_m3
            ) > 1.0e-6
        ):

            raise RuntimeError(
                f"Day {day_index}: "
                "gross signed mapping failed."
            )


        # ----------------------------------------------------
        # Advance connected MF6 one day
        # ----------------------------------------------------

        mf6.prepare_time_step(
            DT_DAYS
        )


        nbound[0] = (
            NODES
        )


        nodelist[:NODES] = np.arange(
            1,
            NODES + 1,
            dtype=
                nodelist.dtype,
        )


        hcof[:NODES] = 0.0


        # Q > 0 = into MF6.
        #
        # Q = HCOF*h - RHS
        #
        # HCOF=0 -> RHS=-Q
        rhs[:NODES] = (
            -interface_m3 /
            DT_DAYS
        )


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

            # Reapply API stress before each solve.
            nbound[0] = (
                NODES
            )


            nodelist[:NODES] = np.arange(
                1,
                NODES + 1,
                dtype=
                    nodelist.dtype,
            )


            hcof[:NODES] = 0.0


            rhs[:NODES] = (
                -interface_m3 /
                DT_DAYS
            )


            converged = mf6.solve(
                component
            )


            iterations = (
                iteration
            )


            if converged:

                break


        if not converged:

            raise RuntimeError(
                f"MF6 failed on "
                f"day {day_index}."
            )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # H8b proved FLOWJA and SIMVALS must be sampled AFTER
        # finalize_solve().
        # ----------------------------------------------------

        mf6.finalize_solve(
            component
        )


        mf6_head_end = np.array(
            head,
            dtype=float,
            copy=True,
        )


        flowja_day = np.array(
            flowja,
            dtype=float,
            copy=True,
        )


        api_flow_m3_day = np.array(
            simvals[:NODES],
            dtype=float,
            copy=True,
        )


        # ----------------------------------------------------
        # Decode lateral flow
        # ----------------------------------------------------

        (
            lateral_rate_m3_day,
            pair_error,
        ) = decode_lateral_flow(
            flowja_day
        )


        lateral_volume_m3 = (
            lateral_rate_m3_day *
            DT_DAYS
        )


        net_lateral_m3 = float(
            lateral_volume_m3.sum()
        )


        # Useful measure of total internal redistribution.
        #
        # Each physical connection appears twice, so divide
        # summed absolute directed cell flow by two.
        gross_lateral_m3 = float(
            0.5 *
            np.sum(
                np.abs(
                    lateral_volume_m3
                )
            )
        )


        # ----------------------------------------------------
        # Storage and full cell budget
        # ----------------------------------------------------

        head_change_m = (
            mf6_head_end -
            mf6_head_start
        )


        storage_change_m3 = (
            head_change_m *
            storage_coeff_m2
        )


        cell_budget_residual_m3 = (
            storage_change_m3 -
            interface_m3 -
            lateral_volume_m3
        )


        max_day_cell_budget_error = float(
            np.max(
                np.abs(
                    cell_budget_residual_m3
                )
            )
        )


        api_error_m3_day = (
            api_flow_m3_day -
            interface_m3 /
            DT_DAYS
        )


        max_day_api_error = float(
            np.max(
                np.abs(
                    api_error_m3_day
                )
            )
        )


        storage_total_m3 = float(
            storage_change_m3.sum()
        )


        domain_storage_interface_error_m3 = (
            storage_total_m3 -
            vic_total_m3
        )


        full_domain_error_m3 = (
            storage_total_m3 -
            vic_total_m3 -
            net_lateral_m3
        )


        # ----------------------------------------------------
        # Reverse MF6 -> VIC map for next coupling step
        # ----------------------------------------------------

        vic_head_end = (
            A.dot(
                mf6_head_end
            ) /
            vic_area_m2
        )


        vic_head_change = (
            vic_head_end -
            vic_head_start
        )


        # ----------------------------------------------------
        # Day-1 regression against H8b
        # ----------------------------------------------------

        if day_index == 1:

            interface_diff = float(
                np.max(
                    np.abs(
                        interface_m3 -
                        h8b_interface_m3
                    )
                )
            )


            lateral_diff = float(
                np.max(
                    np.abs(
                        lateral_volume_m3 -
                        h8b_lateral_m3
                    )
                )
            )


            head_diff = float(
                np.max(
                    np.abs(
                        mf6_head_end -
                        h8b_head_after_m
                    )
                )
            )


            storage_diff = float(
                np.max(
                    np.abs(
                        storage_change_m3 -
                        h8b_storage_m3
                    )
                )
            )


            day1_regression_pass = (
                interface_diff <
                1.0e-6 and

                lateral_diff <
                1.0e-6 and

                head_diff <
                1.0e-12 and

                storage_diff <
                1.0e-6
            )


            print()
            print(
                "H8b DAY-1 REGRESSION"
            )

            print(
                "  interface max diff =",
                interface_diff,
                "m3"
            )

            print(
                "  lateral max diff   =",
                lateral_diff,
                "m3"
            )

            print(
                "  head max diff      =",
                head_diff,
                "m"
            )

            print(
                "  storage max diff   =",
                storage_diff,
                "m3"
            )

            print(
                "  result             =",
                "PASS"
                if day1_regression_pass
                else "FAIL"
            )


        # ----------------------------------------------------
        # Save daily and cellwise diagnostics
        # ----------------------------------------------------

        cumulative_vic_m3 += (
            vic_total_m3
        )


        max_cell_budget_error = max(
            max_cell_budget_error,
            max_day_cell_budget_error,
        )


        max_domain_error = max(
            max_domain_error,
            abs(
                full_domain_error_m3
            ),
        )


        max_api_error = max(
            max_api_error,
            max_day_api_error,
        )


        max_pair_error = max(
            max_pair_error,
            pair_error,
        )


        max_net_lateral = max(
            max_net_lateral,
            abs(
                net_lateral_m3
            ),
        )


        max_water_error = max(
            max_water_error,
            vic_result[
                "water_error_absmax_mm"
            ],
        )


        daily_rows.append({
            "day":
                day_index,

            "date":
                current_date.isoformat(),

            "mf6_iterations":
                iterations,

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

            "mapped_interface_volume_m3":
                mapped_total_m3,

            "mapping_error_m3":
                mapping_error_m3,

            "gross_lateral_redistribution_m3":
                gross_lateral_m3,

            "net_lateral_volume_m3":
                net_lateral_m3,

            "max_pair_antisymmetry_m3_day":
                pair_error,

            "mf6_storage_change_m3":
                storage_total_m3,

            "domain_storage_interface_error_m3":
                domain_storage_interface_error_m3,

            "full_domain_budget_error_m3":
                full_domain_error_m3,

            "max_cell_budget_residual_m3":
                max_day_cell_budget_error,

            "max_api_error_m3_day":
                max_day_api_error,

            "mf6_head_end_min_m":
                float(
                    mf6_head_end.min()
                ),

            "mf6_head_end_max_m":
                float(
                    mf6_head_end.max()
                ),

            "vic_head_end_min_m":
                float(
                    vic_head_end.min()
                ),

            "vic_head_end_max_m":
                float(
                    vic_head_end.max()
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
                        "bottom_moist_mm"
                    ].min()
                ),

            "bottom_moist_max_mm":
                float(
                    vic_result[
                        "bottom_moist_mm"
                    ].max()
                ),

            "vic_water_error_absmax_mm":
                vic_result[
                    "water_error_absmax_mm"
                ],

            "cumulative_vic_exchange_m3":
                cumulative_vic_m3,

            "vic_state_file":
                str(
                    vic_result[
                        "state_file"
                    ]
                ),
        })


        for j in range(
            NODES
        ):

            cell_rows.append({
                "day":
                    day_index,

                "date":
                    current_date.isoformat(),

                "mf6_id":
                    j,

                "head_start_m":
                    mf6_head_start[j],

                "interface_volume_m3":
                    interface_m3[j],

                "lateral_volume_m3":
                    lateral_volume_m3[j],

                "storage_change_m3":
                    storage_change_m3[j],

                "cell_budget_residual_m3":
                    cell_budget_residual_m3[j],

                "head_end_m":
                    mf6_head_end[j],

                "api_simvals_m3_day":
                    api_flow_m3_day[j],
            })


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
            "VIC/interface V   =",
            vic_total_m3,
            "m3"
        )

        print(
            "gross lateral V   =",
            gross_lateral_m3,
            "m3/day-equivalent"
        )

        print(
            "net lateral V     =",
            net_lateral_m3,
            "m3"
        )

        print(
            "MF6 storage       =",
            storage_total_m3,
            "m3"
        )

        print(
            "cell budget max   =",
            max_day_cell_budget_error,
            "m3"
        )

        print(
            "domain error      =",
            full_domain_error_m3,
            "m3"
        )

        print(
            "MF6 end heads     =",
            mf6_head_end.min(),
            mf6_head_end.max(),
            "m"
        )


        # Accepted VIC restart.
        accepted_vic_state = (
            vic_result[
                "state_file"
            ]
        )


        # Advance MF6 physical time exactly once.
        mf6.finalize_time_step()


    final_mf6_head_m = np.array(
        head,
        dtype=float,
        copy=True,
    )


finally:

    mf6.finalize()


# ============================================================
# Cumulative domain conservation
# ============================================================

cumulative_mf6_storage_m3 = float(
    np.sum(
        (
            final_mf6_head_m -
            initial_head_actual
        ) *
        storage_coeff_m2
    )
)


cumulative_error_m3 = (
    cumulative_mf6_storage_m3 -
    cumulative_vic_m3
)


relative_cumulative_error = (
    cumulative_error_m3 /
    cumulative_vic_m3
)


final_vic_head_m = (
    A.dot(
        final_mf6_head_m
    ) /
    vic_area_m2
)


# ============================================================
# Save final VIC head field
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


    for i in range(
        NV
    ):

        f.write(
            f"{vic_lat[i]:.17g} "
            f"{vic_lon[i]:.17g} "
            f"{final_vic_head_m[i]:.17g}\n"
        )


# ============================================================
# Save CSV
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
    "cell_daily.csv"
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
# Validation
# ============================================================

DAY1_PASS = (
    day1_regression_pass
)


PAIR_PASS = (
    max_pair_error <
    1.0e-6
)


LATERAL_DOMAIN_PASS = (
    max_net_lateral <
    1.0e-6
)


API_PASS = (
    max_api_error <
    1.0e-6
)


CELL_BUDGET_PASS = (
    max_cell_budget_error <
    1.0e-2
)


DAILY_DOMAIN_PASS = (
    max_domain_error <
    1.0e-3
)


CUMULATIVE_PASS = (
    abs(
        cumulative_error_m3
    ) <
    1.0e-3
)


WATER_BALANCE_PASS = (
    max_water_error <
    1.0e-8
)


LATERAL_ACTIVE_PASS = (
    max(
        x[
            "gross_lateral_redistribution_m3"
        ]
        for x in daily_rows
    ) >
    1.0e-6
)


OVERALL = all([
    DAY1_PASS,
    PAIR_PASS,
    LATERAL_DOMAIN_PASS,
    API_PASS,
    CELL_BUDGET_PASS,
    DAILY_DOMAIN_PASS,
    CUMULATIVE_PASS,
    WATER_BALANCE_PASS,
    LATERAL_ACTIVE_PASS,
])


summary = {
    "days":
        NDAYS,

    "active_vic_cells":
        NV,

    "mf6_cells":
        NODES,

    "NJA":
        NJA,

    "overlap_records":
        int(
            np.count_nonzero(A)
        ),

    "Ss_per_m":
        SS_PER_M,

    "initial_mf6_head_min_m":
        float(
            initial_head_actual.min()
        ),

    "initial_mf6_head_max_m":
        float(
            initial_head_actual.max()
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

    "cumulative_error_m3":
        cumulative_error_m3,

    "relative_cumulative_error":
        relative_cumulative_error,

    "max_pair_antisymmetry_m3_day":
        max_pair_error,

    "max_net_lateral_volume_m3":
        max_net_lateral,

    "max_cell_budget_residual_m3":
        max_cell_budget_error,

    "max_daily_domain_error_m3":
        max_domain_error,

    "max_api_error_m3_day":
        max_api_error,

    "max_vic_water_error_mm":
        max_water_error,

    "day1_h8b_regression_pass":
        DAY1_PASS,

    "pairwise_lateral_pass":
        PAIR_PASS,

    "lateral_domain_pass":
        LATERAL_DOMAIN_PASS,

    "api_pass":
        API_PASS,

    "cell_budget_pass":
        CELL_BUDGET_PASS,

    "daily_domain_pass":
        DAILY_DOMAIN_PASS,

    "cumulative_pass":
        CUMULATIVE_PASS,

    "water_balance_pass":
        WATER_BALANCE_PASS,

    "lateral_active_pass":
        LATERAL_ACTIVE_PASS,

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
# Print final report
# ============================================================

print()
print("=" * 146)
print("H8c 10-DAY CONNECTED SPATIAL VIC-MF6 LOOP")
print("=" * 146)

print(
    f"{'day':>3s} "
    f"{'q min':>9s} "
    f"{'q max':>9s} "
    f"{'+/-':>6s} "
    f"{'V interface':>14s} "
    f"{'gross lateral':>14s} "
    f"{'cell err':>11s} "
    f"{'domain err':>11s} "
    f"{'h min':>11s} "
    f"{'h max':>11s}"
)


for x in daily_rows:

    print(
        f"{x['day']:3d} "
        f"{x['vic_q_min_mm']:9.4f} "
        f"{x['vic_q_max_mm']:9.4f} "
        f"{x['vic_positive_cells']:2d}/"
        f"{x['vic_negative_cells']:<2d} "
        f"{x['vic_net_volume_m3']:14.3f} "
        f"{x['gross_lateral_redistribution_m3']:14.3f} "
        f"{x['max_cell_budget_residual_m3']:11.3e} "
        f"{x['full_domain_budget_error_m3']:11.3e} "
        f"{x['mf6_head_end_min_m']:11.5f} "
        f"{x['mf6_head_end_max_m']:11.5f}"
    )


print()
print("=" * 92)
print("H8c VALIDATION")
print("=" * 92)

print(
    "H8b day-1 regression:",
    "PASS"
    if DAY1_PASS
    else "FAIL"
)

print(
    "pairwise FLOWJA antisymmetry:",
    "PASS"
    if PAIR_PASS
    else "FAIL",
    "max =",
    max_pair_error,
    "m3/day"
)

print(
    "net lateral domain conservation:",
    "PASS"
    if LATERAL_DOMAIN_PASS
    else "FAIL",
    "max =",
    max_net_lateral,
    "m3"
)

print(
    "API interface flows:",
    "PASS"
    if API_PASS
    else "FAIL",
    "max =",
    max_api_error,
    "m3/day"
)

print(
    "per-cell interface+lateral budget:",
    "PASS"
    if CELL_BUDGET_PASS
    else "FAIL",
    "max =",
    max_cell_budget_error,
    "m3"
)

print(
    "daily domain conservation:",
    "PASS"
    if DAILY_DOMAIN_PASS
    else "FAIL",
    "max =",
    max_domain_error,
    "m3"
)

print(
    "VIC water balance:",
    "PASS"
    if WATER_BALANCE_PASS
    else "FAIL",
    "max =",
    max_water_error,
    "mm"
)

print(
    "cumulative conservation:",
    "PASS"
    if CUMULATIVE_PASS
    else "FAIL"
)

print(
    "lateral groundwater flow active:",
    "PASS"
    if LATERAL_ACTIVE_PASS
    else "FAIL"
)


print()
print("===== CUMULATIVE =====")

print(
    "VIC exchange     =",
    cumulative_vic_m3,
    "m3"
)

print(
    "MF6 storage      =",
    cumulative_mf6_storage_m3,
    "m3"
)

print(
    "error            =",
    cumulative_error_m3,
    "m3"
)

print(
    "relative error   =",
    relative_cumulative_error
)


print()
print("===== HEAD FIELD =====")

print(
    "initial MF6 range =",
    initial_head_actual.min(),
    initial_head_actual.max(),
    "m"
)

print(
    "final MF6 range   =",
    final_mf6_head_m.min(),
    final_mf6_head_m.max(),
    "m"
)

print(
    "final VIC range   =",
    final_vic_head_m.min(),
    final_vic_head_m.max(),
    "m"
)


print()
print("=" * 92)

print(
    "H8c CONNECTED MULTIDAY TWO-WAY COUPLING:",
    "PASS"
    if OVERALL
    else "REQUIRES REVIEW"
)

print("=" * 92)


print()
print("Saved:")

print(
    OUT /
    "daily.csv"
)

print(
    OUT /
    "cell_daily.csv"
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
    "Experiment directory =",
    OUT
)
