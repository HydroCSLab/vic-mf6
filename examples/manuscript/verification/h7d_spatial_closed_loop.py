#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
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

ROOT = Path(
    str(RESULT_ROOT)
)

H7B = (
    ROOT /
    "H7b_stehekin_real_mapper"
)

LIBMF6 = Path(
    str(MF6_LIBRARY)
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    ROOT /
    f"H7d_spatial_closed_loop_{STAMP}"
)

VIC_RUN = OUT / "vic"
MF6_DIR = OUT / "mf6"

VIC_RUN.mkdir(
    parents=True,
    exist_ok=True
)

MF6_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Coupling configuration
# ============================================================

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
        f"found {len(vic_rows)}."
    )


NV = len(vic_rows)


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
        f"found {len(mf6_rows)}."
    )


NM = len(mf6_rows)


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


# H7b deliberately created this heterogeneous
# groundwater-head field.
mf6_head_initial_m = np.array(
    [
        float(
            r["head_m"]
        )
        for r in mf6_rows
    ],
    dtype=float,
)


# ============================================================
# Reconstruct actual H7b overlap matrix
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

    exchange_rows = list(
        csv.DictReader(f)
    )


for r in exchange_rows:

    i = int(
        r["vic_id"]
    )

    j = int(
        r["mf6_id"]
    )

    A[i, j] = float(
        r["overlap_area_m2"]
    )


# Verify exact H7b coverage before proceeding.
coverage = A.sum(
    axis=1
)


if not np.allclose(
    coverage,
    vic_area_m2,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "H7b overlap matrix does not cover "
        "the VIC cells."
    )


print()
print("===== H7d DOMAIN =====")

print(
    "active VIC cells =",
    NV
)

print(
    "MF6 cells        =",
    NM
)

print(
    "overlap records  =",
    np.count_nonzero(A)
)

print(
    "VIC area         =",
    vic_area_m2.sum(),
    "m2"
)

print(
    "MF6 full area    =",
    mf6_area_m2.sum(),
    "m2"
)

print(
    "coupled area     =",
    A.sum(),
    "m2"
)


# ============================================================
# MF6 -> VIC initial-head mapping
# ============================================================

vic_head_initial_m = (
    A.dot(
        mf6_head_initial_m
    ) /
    vic_area_m2
)


print()
print(
    "initial MF6 head range =",
    mf6_head_initial_m.min(),
    mf6_head_initial_m.max(),
    "m"
)

print(
    "initial VIC head range =",
    vic_head_initial_m.min(),
    vic_head_initial_m.max(),
    "m"
)


# ============================================================
# Write VIC spatial-head file
# ============================================================

HEAD_FILE_INITIAL = (
    OUT /
    "vic_head_initial.txt"
)


with HEAD_FILE_INITIAL.open(
    "w"
) as f:

    f.write(
        "# latitude longitude head_offset_m\n"
    )


    for i in range(NV):

        f.write(
            f"{vic_lat[i]:.17g} "
            f"{vic_lon[i]:.17g} "
            f"{vic_head_initial_m[i]:.17g}\n"
        )


# ============================================================
# VIC global configuration
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


base = []

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

    base.append(
        line
    )


BASE_TEXT = (
    "\n".join(base) +
    "\n"
)


RESULT_DIR = (
    VIC_RUN /
    "results"
)

STATE_DIR = (
    VIC_RUN /
    "state"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STATE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


GLOBAL_FILE = (
    VIC_RUN /
    "run.global.txt"
)


GLOBAL_FILE.write_text(
    BASE_TEXT +
    f"""
STARTYEAR    1949
STARTMONTH   1
STARTDAY     1
STARTSEC     0

NRECS        24

RESULT_DIR   {RESULT_DIR}

AGGFREQ      NHOURS 24

STATENAME    {STATE_DIR / 'state'}
STATEYEAR    1949
STATEMONTH   1
STATEDAY     2
STATESEC     0
STATE_FORMAT NETCDF4_CLASSIC
"""
)


# ============================================================
# Run VIC with heterogeneous MF6-derived heads
# ============================================================

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
            HEAD_FILE_INITIAL
        ),
})


env.pop(
    "VIC_GW_HEAD_OFFSET_M",
    None,
)


print()
print("===== RUN VIC =====")


with (
    VIC_RUN /
    "stdout.log"
).open("w") as stdout, \
     (
         VIC_RUN /
         "stderr.log"
     ).open("w") as stderr:

    proc = subprocess.run(
        [
            "mpirun",
            "-n",
            "1",
            str(VIC_EXE),
            "-g",
            str(GLOBAL_FILE),
        ],
        cwd=PARAMDIR,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )


if proc.returncode != 0:

    raise RuntimeError(
        "VIC failed; see "
        f"{VIC_RUN / 'stderr.log'}"
    )


flux_files = sorted(
    RESULT_DIR.glob(
        "fluxes*.nc"
    )
)

state_files = sorted(
    STATE_DIR.glob(
        "state*.nc"
    )
)


if len(flux_files) != 1:

    raise RuntimeError(
        "Expected exactly one VIC flux file."
    )


if len(state_files) != 1:

    raise RuntimeError(
        "Expected exactly one VIC state file."
    )


VIC_FLUX = flux_files[0]
VIC_STATE = state_files[0]


print(
    "[OK] flux  =",
    VIC_FLUX
)

print(
    "[OK] state =",
    VIC_STATE
)


# ============================================================
# Read heterogeneous VIC exchange
# ============================================================

with nc.Dataset(
    VIC_FLUX
) as ds:

    q_grid = np.ma.asarray(
        ds[
            "OUT_GW_EXCHANGE"
        ][0],
        dtype=float,
    )

    soil_grid = np.ma.asarray(
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


# Bottom-layer soil moisture.
bottom_grid = (
    soil_grid[-1]
)


bottom_vic_mm = np.array(
    [
        float(
            bottom_grid[
                vic_row[i],
                vic_col[i]
            ]
        )
        for i in range(NV)
    ],
    dtype=float,
)


vic_volume_m3 = (
    q_vic_mm *
    1.0e-3 *
    vic_area_m2
)


vic_total_m3 = float(
    vic_volume_m3.sum()
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


print()
print("===== VIC SPATIAL RESPONSE =====")

print(
    "q range           =",
    q_vic_mm.min(),
    q_vic_mm.max(),
    "mm/day"
)

print(
    "positive cells    =",
    int(
        np.sum(
            q_vic_mm > 0.0
        )
    )
)

print(
    "negative cells    =",
    int(
        np.sum(
            q_vic_mm < 0.0
        )
    )
)

print(
    "total VIC volume  =",
    vic_total_m3,
    "m3"
)

print(
    "positive transfer =",
    vic_positive_m3,
    "m3"
)

print(
    "negative transfer =",
    vic_negative_m3,
    "m3"
)


# ============================================================
# VIC -> MF6 overlap mapping
# ============================================================

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


mf6_total_m3 = float(
    mf6_volume_m3.sum()
)


mapping_error_m3 = (
    mf6_total_m3 -
    vic_total_m3
)


positive_mapping_error_m3 = (
    float(
        mf6_positive_m3.sum()
    ) -
    vic_positive_m3
)


negative_mapping_error_m3 = (
    float(
        mf6_negative_m3.sum()
    ) -
    vic_negative_m3
)


print()
print("===== VIC -> MF6 MAPPING =====")

print(
    "mapped MF6 total =",
    mf6_total_m3,
    "m3"
)

print(
    "net error        =",
    mapping_error_m3,
    "m3"
)

print(
    "positive error   =",
    positive_mapping_error_m3,
    "m3"
)

print(
    "negative error   =",
    negative_mapping_error_m3,
    "m3"
)


if not np.isclose(
    mf6_total_m3,
    vic_total_m3,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "VIC->MF6 net mapping failed."
    )


# ============================================================
# Build actual 12-cell MF6 model
# ============================================================

area_text = " ".join(
    f"{x:.17g}"
    for x in mf6_area_m2
)


head_text = " ".join(
    f"{x:.17g}"
    for x in mf6_head_initial_m
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
  TDIS6 h7d.tdis
END TIMING

BEGIN MODELS
  GWF6 h7d.nam H7D
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h7d.ims H7D
END SOLUTIONGROUP
"""
)


(MF6_DIR / "h7d.tdis").write_text(
"""BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  1.0 1 1.0
END PERIODDATA
"""
)


(MF6_DIR / "h7d.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
)


(MF6_DIR / "h7d.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DISU6 h7d.disu DISU
  IC6   h7d.ic   IC
  NPF6  h7d.npf  NPF
  STO6  h7d.sto  STO
  API6  h7d.api  VICAPI
  OC6   h7d.oc   OC
END PACKAGES
"""
)


(MF6_DIR / "h7d.disu").write_text(
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


(MF6_DIR / "h7d.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    INTERNAL FACTOR 1.0
    {head_text}
END GRIDDATA
"""
)


(MF6_DIR / "h7d.npf").write_text(
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


(MF6_DIR / "h7d.sto").write_text(
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


(MF6_DIR / "h7d.api").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND {NM}
END DIMENSIONS
"""
)


(MF6_DIR / "h7d.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h7d.hds
  BUDGET FILEOUT h7d.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
)


# ============================================================
# Expected isolated-cell storage response
# ============================================================

storage_coeff_m2 = (
    SS_PER_M *
    mf6_area_m2 *
    THICKNESS_M
)


expected_dh_m = (
    mf6_volume_m3 /
    storage_coeff_m2
)


expected_head_final_m = (
    mf6_head_initial_m +
    expected_dh_m
)


# ============================================================
# MF6 API solve
# ============================================================

mf6 = XmiWrapper(
    str(LIBMF6),
    working_directory=
        str(MF6_DIR),
)


mf6.initialize_mpi(MPI.COMM_SELF.py2f())


try:

    head = mf6.get_value_ptr(
        mf6.get_var_address(
            "X",
            "H7D"
        )
    )


    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H7D",
            "VICAPI"
        )
    )


    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H7D",
            "VICAPI"
        )
    )


    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H7D",
            "VICAPI"
        )
    )


    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H7D",
            "VICAPI"
        )
    )


    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H7D",
            "VICAPI"
        )
    )


    actual_head_initial_m = np.array(
        head,
        dtype=float,
        copy=True,
    )


    initial_head_error = (
        actual_head_initial_m -
        mf6_head_initial_m
    )


    print()
    print("===== MF6 INITIALIZATION =====")

    print(
        "max initial-head error =",
        np.max(
            np.abs(
                initial_head_error
            )
        ),
        "m"
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

        mf6_iterations = iteration


        if converged:
            break


    if not converged:

        raise RuntimeError(
            "MF6 H7d solve did not converge."
        )


    mf6.finalize_solve(
        component
    )

    mf6.finalize_time_step()


    mf6_head_final_m = np.array(
        head,
        dtype=float,
        copy=True,
    )


    api_flow_m3_day = np.array(
        simvals[:NM],
        dtype=float,
        copy=True,
    )


finally:

    mf6.finalize()


# ============================================================
# MF6 verification
# ============================================================

actual_dh_m = (
    mf6_head_final_m -
    actual_head_initial_m
)


actual_storage_m3 = (
    actual_dh_m *
    storage_coeff_m2
)


cell_storage_error_m3 = (
    actual_storage_m3 -
    mf6_volume_m3
)


head_error_m = (
    mf6_head_final_m -
    expected_head_final_m
)


api_error_m3_day = (
    api_flow_m3_day -
    mf6_volume_m3 /
    DT_DAYS
)


total_storage_m3 = float(
    actual_storage_m3.sum()
)


cross_model_error_m3 = (
    total_storage_m3 -
    vic_total_m3
)


relative_cross_model_error = (
    cross_model_error_m3 /
    vic_total_m3
)


print()
print("===== MF6 RESPONSE =====")

print(
    "final head range =",
    mf6_head_final_m.min(),
    mf6_head_final_m.max(),
    "m"
)

print(
    "max head error   =",
    np.max(
        np.abs(
            head_error_m
        )
    ),
    "m"
)

print(
    "max cell dV err  =",
    np.max(
        np.abs(
            cell_storage_error_m3
        )
    ),
    "m3"
)

print(
    "max API error    =",
    np.max(
        np.abs(
            api_error_m3_day
        )
    ),
    "m3/day"
)

print(
    "MF6 storage      =",
    total_storage_m3,
    "m3"
)

print(
    "VIC transfer     =",
    vic_total_m3,
    "m3"
)

print(
    "cross-model err  =",
    cross_model_error_m3,
    "m3"
)

print(
    "relative error   =",
    relative_cross_model_error
)


# ============================================================
# MF6 -> VIC updated-head mapping
#
# This closes the spatial loop: the resulting MF6 head field
# is transformed back to the head field that VIC would receive
# at the next coupling interval.
# ============================================================

vic_head_final_m = (
    A.dot(
        mf6_head_final_m
    ) /
    vic_area_m2
)


vic_head_change_m = (
    vic_head_final_m -
    vic_head_initial_m
)


HEAD_FILE_FINAL = (
    OUT /
    "vic_head_after_mf6.txt"
)


with HEAD_FILE_FINAL.open(
    "w"
) as f:

    f.write(
        "# latitude longitude head_offset_m\n"
    )


    for i in range(NV):

        f.write(
            f"{vic_lat[i]:.17g} "
            f"{vic_lon[i]:.17g} "
            f"{vic_head_final_m[i]:.17g}\n"
        )


print()
print("===== RETURN MF6 -> VIC MAPPING =====")

print(
    "updated VIC head range =",
    vic_head_final_m.min(),
    vic_head_final_m.max(),
    "m"
)

print(
    "VIC head-change range  =",
    vic_head_change_m.min(),
    vic_head_change_m.max(),
    "m"
)


# ============================================================
# Save VIC-cell diagnostics
# ============================================================

vic_output = []


for i in range(NV):

    vic_output.append({
        "vic_id":
            i,

        "row":
            vic_row[i],

        "col":
            vic_col[i],

        "lat":
            vic_lat[i],

        "lon":
            vic_lon[i],

        "area_m2":
            vic_area_m2[i],

        "initial_mapped_mf6_head_m":
            vic_head_initial_m[i],

        "gw_exchange_mm":
            q_vic_mm[i],

        "vic_exchange_volume_m3":
            vic_volume_m3[i],

        "bottom_soil_moisture_mm":
            bottom_vic_mm[i],

        "final_mapped_mf6_head_m":
            vic_head_final_m[i],

        "mapped_head_change_m":
            vic_head_change_m[i],
    })


with (
    OUT /
    "vic_cells.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            vic_output[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        vic_output
    )


# ============================================================
# Save MF6-cell diagnostics
# ============================================================

mf6_output = []


for j in range(NM):

    mf6_output.append({
        "mf6_id":
            j,

        "full_area_m2":
            mf6_area_m2[j],

        "coupled_area_m2":
            mf6_coupled_area_m2[j],

        "initial_head_m":
            actual_head_initial_m[j],

        "mapped_positive_volume_m3":
            mf6_positive_m3[j],

        "mapped_negative_volume_m3":
            mf6_negative_m3[j],

        "mapped_net_volume_m3":
            mf6_volume_m3[j],

        "api_simvals_m3_day":
            api_flow_m3_day[j],

        "expected_head_final_m":
            expected_head_final_m[j],

        "actual_head_final_m":
            mf6_head_final_m[j],

        "head_error_m":
            head_error_m[j],

        "actual_storage_change_m3":
            actual_storage_m3[j],

        "storage_error_m3":
            cell_storage_error_m3[j],
    })


with (
    OUT /
    "mf6_cells.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            mf6_output[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        mf6_output
    )


# ============================================================
# Pass/fail
# ============================================================

INITIAL_HEAD_PASS = (
    np.max(
        np.abs(
            initial_head_error
        )
    ) <
    1.0e-12
)


MAPPING_PASS = (
    abs(
        mapping_error_m3
    ) <
    1.0e-6
)


GROSS_MAPPING_PASS = (
    abs(
        positive_mapping_error_m3
    ) <
    1.0e-6
    and
    abs(
        negative_mapping_error_m3
    ) <
    1.0e-6
)


API_PASS = (
    np.max(
        np.abs(
            api_error_m3_day
        )
    ) <
    1.0e-6
)


HEAD_PASS = (
    np.max(
        np.abs(
            head_error_m
        )
    ) <
    1.0e-9
)


CELL_STORAGE_PASS = (
    np.max(
        np.abs(
            cell_storage_error_m3
        )
    ) <
    1.0e-3
)


GLOBAL_STORAGE_PASS = (
    abs(
        cross_model_error_m3
    ) <
    1.0e-3
)


RETURN_HEAD_PASS = bool(
    np.all(
        np.isfinite(
            vic_head_final_m
        )
    )
)


OVERALL = all([
    INITIAL_HEAD_PASS,
    MAPPING_PASS,
    GROSS_MAPPING_PASS,
    API_PASS,
    HEAD_PASS,
    CELL_STORAGE_PASS,
    GLOBAL_STORAGE_PASS,
    RETURN_HEAD_PASS,
])


# ============================================================
# Summary
# ============================================================

summary = {
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

    "mf6_iterations":
        mf6_iterations,

    "initial_mf6_head_min_m":
        float(
            mf6_head_initial_m.min()
        ),

    "initial_mf6_head_max_m":
        float(
            mf6_head_initial_m.max()
        ),

    "initial_vic_head_min_m":
        float(
            vic_head_initial_m.min()
        ),

    "initial_vic_head_max_m":
        float(
            vic_head_initial_m.max()
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

    "vic_total_exchange_m3":
        vic_total_m3,

    "mapped_mf6_total_m3":
        mf6_total_m3,

    "mapping_error_m3":
        mapping_error_m3,

    "positive_mapping_error_m3":
        positive_mapping_error_m3,

    "negative_mapping_error_m3":
        negative_mapping_error_m3,

    "mf6_storage_change_m3":
        total_storage_m3,

    "cross_model_error_m3":
        cross_model_error_m3,

    "relative_cross_model_error":
        relative_cross_model_error,

    "max_api_error_m3_day":
        float(
            np.max(
                np.abs(
                    api_error_m3_day
                )
            )
        ),

    "max_mf6_head_error_m":
        float(
            np.max(
                np.abs(
                    head_error_m
                )
            )
        ),

    "max_cell_storage_error_m3":
        float(
            np.max(
                np.abs(
                    cell_storage_error_m3
                )
            )
        ),

    "final_mf6_head_min_m":
        float(
            mf6_head_final_m.min()
        ),

    "final_mf6_head_max_m":
        float(
            mf6_head_final_m.max()
        ),

    "final_vic_head_min_m":
        float(
            vic_head_final_m.min()
        ),

    "final_vic_head_max_m":
        float(
            vic_head_final_m.max()
        ),

    "vic_head_change_min_m":
        float(
            vic_head_change_m.min()
        ),

    "vic_head_change_max_m":
        float(
            vic_head_change_m.max()
        ),

    "vic_water_error_absmax_mm":
        float(
            np.max(
                np.abs(
                    water_error.compressed()
                )
            )
        ),

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
# Print final result
# ============================================================

print()
print("=" * 112)
print("H7d ONE-STEP SPATIAL TWO-WAY COUPLING")
print("=" * 112)

print(
    f"{'test':<34s} {'result':<8s}"
)

print("-" * 48)


tests = [
    (
        "heterogeneous MF6 initialization",
        INITIAL_HEAD_PASS,
    ),
    (
        "VIC -> MF6 net mapping",
        MAPPING_PASS,
    ),
    (
        "gross signed mapping",
        GROSS_MAPPING_PASS,
    ),
    (
        "MF6 API flow",
        API_PASS,
    ),
    (
        "MF6 per-cell head response",
        HEAD_PASS,
    ),
    (
        "MF6 per-cell storage",
        CELL_STORAGE_PASS,
    ),
    (
        "global VIC/MF6 conservation",
        GLOBAL_STORAGE_PASS,
    ),
    (
        "MF6 -> VIC return head map",
        RETURN_HEAD_PASS,
    ),
]


for name, passed in tests:

    print(
        f"{name:<34s} "
        f"{'PASS' if passed else 'FAIL':<8s}"
    )


print()
print(
    "VIC transfer            =",
    vic_total_m3,
    "m3"
)

print(
    "MF6 mapped transfer     =",
    mf6_total_m3,
    "m3"
)

print(
    "MF6 storage response    =",
    total_storage_m3,
    "m3"
)

print(
    "cross-model error       =",
    cross_model_error_m3,
    "m3"
)

print(
    "relative error          =",
    relative_cross_model_error
)

print()
print(
    "initial VIC head range  =",
    vic_head_initial_m.min(),
    vic_head_initial_m.max(),
    "m"
)

print(
    "final VIC head range    =",
    vic_head_final_m.min(),
    vic_head_final_m.max(),
    "m"
)

print(
    "mapped head change      =",
    vic_head_change_m.min(),
    vic_head_change_m.max(),
    "m"
)

print()
print(
    "VIC q range             =",
    q_vic_mm.min(),
    q_vic_mm.max(),
    "mm/day"
)

print(
    "positive/negative cells =",
    int(np.sum(q_vic_mm > 0.0)),
    "/",
    int(np.sum(q_vic_mm < 0.0))
)

print()
print("=" * 112)

print(
    "H7d SPATIAL TWO-WAY LOOP:",
    "PASS"
    if OVERALL
    else "REQUIRES REVIEW"
)

print("=" * 112)


print()
print("Saved:")
print(OUT / "vic_head_initial.txt")
print(OUT / "vic_head_after_mf6.txt")
print(OUT / "vic_cells.csv")
print(OUT / "mf6_cells.csv")
print(OUT / "summary.csv")
print(VIC_STATE)

print()
print("Experiment directory:")
print(OUT)
