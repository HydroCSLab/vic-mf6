#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv
import time

import numpy as np
from xmipy import XmiWrapper
from mpi4py import MPI


# ============================================================
# Paths
# ============================================================

VIC_ROOT = Path(
    str(VIC_ROOT)
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

MF6_CSV = (
    H7B /
    "mf6_cells.csv"
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    ROOT /
    f"H7c_spatial_api_12cell_{STAMP}"
)

MF6_DIR = (
    OUT /
    "mf6"
)

MF6_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experimental properties
# ============================================================

INITIAL_HEAD_M = -105.328

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)

SS_PER_M = 1.0e-5

DT_DAYS = 1.0


# ============================================================
# Read mapped H7b MF6 cells
# ============================================================

with MF6_CSV.open() as f:

    rows = list(
        csv.DictReader(f)
    )


rows.sort(
    key=lambda r:
        int(r["mf6_id"])
)


if len(rows) != 12:

    raise RuntimeError(
        f"Expected 12 H7b MF6 cells; "
        f"found {len(rows)}."
    )


ids = np.array(
    [
        int(r["mf6_id"])
        for r in rows
    ],
    dtype=int,
)


if not np.array_equal(
    ids,
    np.arange(12),
):

    raise RuntimeError(
        f"Unexpected MF6 ids: {ids}"
    )


# Full conceptual MF6 cell area.
#
# Do NOT use coupled_area_m2 for storage.
# The exchange applies only over the VIC-overlap area,
# but groundwater storage belongs to the complete MF6 cell.

area_m2 = np.array(
    [
        float(
            r["rectangle_area_m2"]
        )
        for r in rows
    ],
    dtype=float,
)


coupled_area_m2 = np.array(
    [
        float(
            r["coupled_area_m2"]
        )
        for r in rows
    ],
    dtype=float,
)


mapped_volume_m3 = np.array(
    [
        float(
            r["mapped_net_volume_m3"]
        )
        for r in rows
    ],
    dtype=float,
)


positive_volume_m3 = np.array(
    [
        float(
            r[
                "mapped_positive_volume_m3"
            ]
        )
        for r in rows
    ],
    dtype=float,
)


negative_volume_m3 = np.array(
    [
        float(
            r[
                "mapped_negative_volume_m3"
            ]
        )
        for r in rows
    ],
    dtype=float,
)


# H7b values are one-day transfer volumes.
Q_m3_day = (
    mapped_volume_m3 /
    DT_DAYS
)


storage_coeff_m2 = (
    SS_PER_M *
    area_m2 *
    THICKNESS_M
)


expected_dh_m = (
    mapped_volume_m3 /
    storage_coeff_m2
)


expected_head_m = (
    INITIAL_HEAD_M +
    expected_dh_m
)


print()
print("===== H7c INPUT =====")

print(
    "cells                  =",
    len(rows)
)

print(
    "Ss                     =",
    SS_PER_M,
    "1/m"
)

print(
    "thickness              =",
    THICKNESS_M,
    "m"
)

print(
    "full MF6 area          =",
    area_m2.sum(),
    "m2"
)

print(
    "VIC-coupled area       =",
    coupled_area_m2.sum(),
    "m2"
)

print(
    "mapped VIC volume      =",
    mapped_volume_m3.sum(),
    "m3"
)

print(
    "mapped positive volume =",
    positive_volume_m3.sum(),
    "m3"
)

print(
    "mapped negative volume =",
    negative_volume_m3.sum(),
    "m3"
)


# ============================================================
# Write one real 12-node GWF model.
#
# DISU is deliberately disconnected:
#
#   NODES = 12
#   NJA   = 12
#   IAC   = 1 for every cell
#   JA    = self only
#
# Thus there is no lateral groundwater exchange in H7c.
# That is intentional: this test isolates spatial API
# transfer and per-cell storage.
# ============================================================

(MF6_DIR / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h7c.tdis
END TIMING

BEGIN MODELS
  GWF6 h7c.nam H7C
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h7c.ims H7C
END SOLUTIONGROUP
"""
)


(MF6_DIR / "h7c.tdis").write_text(
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


(MF6_DIR / "h7c.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
)


(MF6_DIR / "h7c.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DISU6 h7c.disu DISU
  IC6   h7c.ic   IC
  NPF6  h7c.npf  NPF
  STO6  h7c.sto  STO
  API6  h7c.api  VICAPI
  OC6   h7c.oc   OC
END PACKAGES
"""
)


# ------------------------------------------------------------
# DISU
# ------------------------------------------------------------

area_text = " ".join(
    f"{x:.17g}"
    for x in area_m2
)

iac_text = " ".join(
    "1"
    for _ in rows
)

ja_text = " ".join(
    str(i)
    for i in range(
        1,
        len(rows) + 1
    )
)


(MF6_DIR / "h7c.disu").write_text(
f"""BEGIN OPTIONS
  LENGTH_UNITS METERS
END OPTIONS

BEGIN DIMENSIONS
  NODES 12
  NJA 12
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


# ------------------------------------------------------------
# Initial condition
# ------------------------------------------------------------

(MF6_DIR / "h7c.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    CONSTANT {INITIAL_HEAD_M:.17g}
END GRIDDATA
"""
)


# ------------------------------------------------------------
# NPF
#
# There are no off-diagonal DISU connections, so K cannot
# generate lateral flow. NPF is retained to keep this a
# normal GWF model configuration.
# ------------------------------------------------------------

(MF6_DIR / "h7c.npf").write_text(
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


# ------------------------------------------------------------
# STO
# ------------------------------------------------------------

(MF6_DIR / "h7c.sto").write_text(
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


# ------------------------------------------------------------
# API boundary
# ------------------------------------------------------------

(MF6_DIR / "h7c.api").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND 12
END DIMENSIONS
"""
)


# ------------------------------------------------------------
# Output control
# ------------------------------------------------------------

(MF6_DIR / "h7c.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h7c.hds
  BUDGET FILEOUT h7c.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
)


# ============================================================
# Run through XMI
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
            "H7C"
        )
    )


    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H7C",
            "VICAPI"
        )
    )


    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H7C",
            "VICAPI"
        )
    )


    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H7C",
            "VICAPI"
        )
    )


    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H7C",
            "VICAPI"
        )
    )


    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H7C",
            "VICAPI"
        )
    )


    print()
    print("===== XMI ARRAY SHAPES =====")

    print(
        "X        ",
        head.shape
    )

    print(
        "NBOUND   ",
        nbound.shape
    )

    print(
        "NODELIST ",
        nodelist.shape
    )

    print(
        "RHS      ",
        rhs.shape
    )

    print(
        "HCOF     ",
        hcof.shape
    )

    print(
        "SIMVALS  ",
        simvals.shape
    )


    if head.size != 12:

        raise RuntimeError(
            f"Expected 12 heads; got {head.size}"
        )


    # ========================================================
    # Prepare one-day MF6 step
    # ========================================================

    head_before = np.array(
        head,
        dtype=float,
        copy=True,
    )


    mf6.prepare_time_step(
        DT_DAYS
    )


    # Sign convention established in H2:
    #
    #   Q > 0 : water enters MF6
    #
    # API equation:
    #
    #   Qcell = HCOF*h - RHS
    #
    # For pure specified flux:
    #
    #   HCOF = 0
    #   RHS  = -Q

    nbound[0] = 12

    nodelist[:12] = np.arange(
        1,
        13,
        dtype=nodelist.dtype,
    )

    hcof[:12] = 0.0

    rhs[:12] = (
        -Q_m3_day
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

        # Reapply API values because package arrays can be
        # touched during formulation/solve stages.

        nbound[0] = 12

        nodelist[:12] = np.arange(
            1,
            13,
            dtype=nodelist.dtype,
        )

        hcof[:12] = 0.0

        rhs[:12] = (
            -Q_m3_day
        )


        converged = mf6.solve(
            component
        )

        iterations = iteration


        if converged:
            break


    if not converged:

        raise RuntimeError(
            "H7c MF6 solve did not converge."
        )


    mf6.finalize_solve(
        component
    )

    mf6.finalize_time_step()


    head_after = np.array(
        head,
        dtype=float,
        copy=True,
    )


    api_flow = np.array(
        simvals[:12],
        dtype=float,
        copy=True,
    )


finally:

    mf6.finalize()


# ============================================================
# Per-cell verification
# ============================================================

dh_m = (
    head_after -
    head_before
)


storage_volume_m3 = (
    dh_m *
    storage_coeff_m2
)


head_error_m = (
    head_after -
    expected_head_m
)


api_flow_error_m3_day = (
    api_flow -
    Q_m3_day
)


storage_error_m3 = (
    storage_volume_m3 -
    mapped_volume_m3
)


results = []


for j in range(12):

    mapped = (
        mapped_volume_m3[j]
    )


    if mapped != 0.0:

        relative_storage_error = (
            storage_error_m3[j] /
            mapped
        )

    else:

        relative_storage_error = 0.0


    results.append({
        "mf6_id":
            j,

        "node":
            j + 1,

        "full_area_m2":
            area_m2[j],

        "coupled_area_m2":
            coupled_area_m2[j],

        "coupled_fraction":
            coupled_area_m2[j] /
            area_m2[j],

        "mapped_positive_m3":
            positive_volume_m3[j],

        "mapped_negative_m3":
            negative_volume_m3[j],

        "mapped_net_m3":
            mapped_volume_m3[j],

        "api_Q_m3_day":
            Q_m3_day[j],

        "api_SIMVALS_m3_day":
            api_flow[j],

        "api_flow_error_m3_day":
            api_flow_error_m3_day[j],

        "storage_coefficient_m2":
            storage_coeff_m2[j],

        "head_before_m":
            head_before[j],

        "expected_dh_m":
            expected_dh_m[j],

        "actual_dh_m":
            dh_m[j],

        "expected_head_m":
            expected_head_m[j],

        "actual_head_m":
            head_after[j],

        "head_error_m":
            head_error_m[j],

        "expected_storage_volume_m3":
            mapped_volume_m3[j],

        "actual_storage_volume_m3":
            storage_volume_m3[j],

        "storage_error_m3":
            storage_error_m3[j],

        "relative_storage_error":
            relative_storage_error,
    })


# ============================================================
# Pass/fail metrics
# ============================================================

max_head_error = float(
    np.max(
        np.abs(
            head_error_m
        )
    )
)


max_api_error = float(
    np.max(
        np.abs(
            api_flow_error_m3_day
        )
    )
)


max_storage_error = float(
    np.max(
        np.abs(
            storage_error_m3
        )
    )
)


total_mapped = float(
    mapped_volume_m3.sum()
)


total_storage = float(
    storage_volume_m3.sum()
)


total_error = (
    total_storage -
    total_mapped
)


relative_total_error = (
    total_error /
    total_mapped
)


positive_storage = float(
    np.sum(
        storage_volume_m3[
            mapped_volume_m3 > 0.0
        ]
    )
)


negative_storage = float(
    np.sum(
        storage_volume_m3[
            mapped_volume_m3 < 0.0
        ]
    )
)


expected_positive = float(
    np.sum(
        mapped_volume_m3[
            mapped_volume_m3 > 0.0
        ]
    )
)


expected_negative = float(
    np.sum(
        mapped_volume_m3[
            mapped_volume_m3 < 0.0
        ]
    )
)


HEAD_PASS = (
    max_head_error <
    1.0e-9
)

API_PASS = (
    max_api_error <
    1.0e-6
)

CELL_VOLUME_PASS = (
    max_storage_error <
    1.0e-3
)

TOTAL_VOLUME_PASS = (
    abs(total_error) <
    1.0e-3
)


# ============================================================
# Save results
# ============================================================

with (
    OUT /
    "cell_results.csv"
).open(
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


summary = {
    "cells":
        12,

    "mf6_iterations":
        iterations,

    "Ss_per_m":
        SS_PER_M,

    "thickness_m":
        THICKNESS_M,

    "total_full_mf6_area_m2":
        float(
            area_m2.sum()
        ),

    "total_coupled_vic_area_m2":
        float(
            coupled_area_m2.sum()
        ),

    "mapped_total_volume_m3":
        total_mapped,

    "mf6_total_storage_change_m3":
        total_storage,

    "total_volume_error_m3":
        total_error,

    "relative_total_volume_error":
        relative_total_error,

    "max_cell_storage_error_m3":
        max_storage_error,

    "max_head_error_m":
        max_head_error,

    "max_api_flow_error_m3_day":
        max_api_error,

    "expected_positive_net_cell_volume_m3":
        expected_positive,

    "actual_positive_storage_change_m3":
        positive_storage,

    "expected_negative_net_cell_volume_m3":
        expected_negative,

    "actual_negative_storage_change_m3":
        negative_storage,

    "head_test_pass":
        HEAD_PASS,

    "api_flow_test_pass":
        API_PASS,

    "cell_volume_test_pass":
        CELL_VOLUME_PASS,

    "total_volume_test_pass":
        TOTAL_VOLUME_PASS,
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
# Print
# ============================================================

print()
print("=" * 122)
print("H7c SPATIAL VIC -> 12-CELL MF6 API TEST")
print("=" * 122)

print(
    f"{'id':>3s} "
    f"{'mapped V':>14s} "
    f"{'Q API':>14s} "
    f"{'h expected':>14s} "
    f"{'h actual':>14s} "
    f"{'dh err':>12s} "
    f"{'storage V':>14s} "
    f"{'dV':>12s}"
)


for r in results:

    print(
        f"{r['mf6_id']:3d} "
        f"{r['mapped_net_m3']:14.3f} "
        f"{r['api_SIMVALS_m3_day']:14.3f} "
        f"{r['expected_head_m']:14.8f} "
        f"{r['actual_head_m']:14.8f} "
        f"{r['head_error_m']:12.3e} "
        f"{r['actual_storage_volume_m3']:14.3f} "
        f"{r['storage_error_m3']:12.3e}"
    )


print()
print("=" * 72)
print("H7c TESTS")
print("=" * 72)

print(
    "PER-CELL HEAD:",
    "PASS"
    if HEAD_PASS
    else "FAIL",
    "max error =",
    max_head_error,
    "m"
)

print(
    "API FLOWS:",
    "PASS"
    if API_PASS
    else "FAIL",
    "max error =",
    max_api_error,
    "m3/day"
)

print(
    "PER-CELL STORAGE:",
    "PASS"
    if CELL_VOLUME_PASS
    else "FAIL",
    "max error =",
    max_storage_error,
    "m3"
)

print(
    "TOTAL STORAGE:",
    "PASS"
    if TOTAL_VOLUME_PASS
    else "FAIL"
)

print()
print(
    "mapped VIC volume =",
    total_mapped,
    "m3"
)

print(
    "MF6 storage change =",
    total_storage,
    "m3"
)

print(
    "total error        =",
    total_error,
    "m3"
)

print(
    "relative error     =",
    relative_total_error
)

print()
print(
    "expected positive net-cell transfer =",
    expected_positive,
    "m3"
)

print(
    "actual positive storage change      =",
    positive_storage,
    "m3"
)

print(
    "expected negative net-cell transfer =",
    expected_negative,
    "m3"
)

print(
    "actual negative storage change      =",
    negative_storage,
    "m3"
)


OVERALL = (
    HEAD_PASS and
    API_PASS and
    CELL_VOLUME_PASS and
    TOTAL_VOLUME_PASS
)


print()
print("=" * 72)

print(
    "H7c SPATIAL API TRANSFER:",
    "PASS"
    if OVERALL
    else "REQUIRES REVIEW"
)

print("=" * 72)

print()
print("Saved:")
print(OUT / "cell_results.csv")
print(OUT / "summary.csv")

print()
print("Experiment directory:")
print(OUT)
