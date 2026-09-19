#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv
import shutil
import time

import numpy as np
from xmipy import XmiWrapper
from mpi4py import MPI


# ============================================================
# Paths
# ============================================================

LIBMF6 = Path(
    str(MF6_LIBRARY)
)

ROOT = Path(
    str(RESULT_ROOT)
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    ROOT /
    f"H8b_connected_interface_budget_{STAMP}"
)

MF6_DIR = OUT / "mf6"

MF6_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Locate validated H7d and H8a experiments
# ============================================================

h7d_runs = sorted(
    ROOT.glob(
        "H7d_spatial_closed_loop_*"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)

h8a_runs = sorted(
    ROOT.glob(
        "H8a_connected_mf6_lateral_*"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)


if not h7d_runs:
    raise RuntimeError(
        "No H7d experiment found."
    )


if not h8a_runs:
    raise RuntimeError(
        "No H8a experiment found."
    )


H7D = h7d_runs[0]
H8A = h8a_runs[0]

H8A_MF6 = (
    H8A /
    "mf6"
)


print()
print("===== H8b INPUT =====")

print(
    "H7d interface source =",
    H7D
)

print(
    "H8a connected model  =",
    H8A
)


# ============================================================
# Configuration
# ============================================================

NROW = 3
NCOL = 4

NODES = (
    NROW *
    NCOL
)

DT_DAYS = 1.0

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)

SS_PER_M = 1.0e-5


# ============================================================
# Read H7d MF6 interface quantities
# ============================================================

with (
    H7D /
    "mf6_cells.csv"
).open() as f:

    h7d_mf6 = list(
        csv.DictReader(f)
    )


h7d_mf6.sort(
    key=lambda r:
        int(r["mf6_id"])
)


if len(h7d_mf6) != NODES:

    raise RuntimeError(
        f"Expected {NODES} H7d MF6 cells; "
        f"found {len(h7d_mf6)}."
    )


area_m2 = np.array(
    [
        float(
            r["full_area_m2"]
        )
        for r in h7d_mf6
    ],
    dtype=float,
)


initial_head_m = np.array(
    [
        float(
            r["initial_head_m"]
        )
        for r in h7d_mf6
    ],
    dtype=float,
)


interface_volume_m3 = np.array(
    [
        float(
            r["mapped_net_volume_m3"]
        )
        for r in h7d_mf6
    ],
    dtype=float,
)


interface_positive_m3 = np.array(
    [
        float(
            r["mapped_positive_volume_m3"]
        )
        for r in h7d_mf6
    ],
    dtype=float,
)


interface_negative_m3 = np.array(
    [
        float(
            r["mapped_negative_volume_m3"]
        )
        for r in h7d_mf6
    ],
    dtype=float,
)


h7d_disconnected_final_head_m = np.array(
    [
        float(
            r["actual_head_final_m"]
        )
        for r in h7d_mf6
    ],
    dtype=float,
)


# ============================================================
# Independently check H7d VIC total
# ============================================================

with (
    H7D /
    "vic_cells.csv"
).open() as f:

    h7d_vic = list(
        csv.DictReader(f)
    )


vic_total_m3 = float(
    np.sum(
        [
            float(
                r[
                    "vic_exchange_volume_m3"
                ]
            )
            for r in h7d_vic
        ]
    )
)


interface_total_m3 = float(
    interface_volume_m3.sum()
)


if not np.isclose(
    vic_total_m3,
    interface_total_m3,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "H7d VIC and mapped MF6 volumes "
        "do not agree."
    )


print()
print(
    "VIC/interface total =",
    interface_total_m3,
    "m3"
)

print(
    "positive gross      =",
    interface_positive_m3.sum(),
    "m3"
)

print(
    "negative gross      =",
    interface_negative_m3.sum(),
    "m3"
)


# ============================================================
# Reuse EXACT validated H8a connected DISU model
# ============================================================

for src_name, dst_name in [
    (
        "h8a.disu",
        "h8b.disu",
    ),
    (
        "h8a.ic",
        "h8b.ic",
    ),
    (
        "h8a.npf",
        "h8b.npf",
    ),
    (
        "h8a.sto",
        "h8b.sto",
    ),
]:

    shutil.copyfile(
        H8A_MF6 /
        src_name,

        MF6_DIR /
        dst_name,
    )


# Solver can also be reused unchanged.
shutil.copyfile(
    H8A_MF6 /
    "h8a.ims",

    MF6_DIR /
    "h8b.ims",
)


# ============================================================
# One one-day connected coupling step
# ============================================================

(MF6_DIR / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h8b.tdis
END TIMING

BEGIN MODELS
  GWF6 h8b.nam H8B
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h8b.ims H8B
END SOLUTIONGROUP
"""
)


(MF6_DIR / "h8b.tdis").write_text(
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


(MF6_DIR / "h8b.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DISU6 h8b.disu DISU
  IC6   h8b.ic   IC
  NPF6  h8b.npf  NPF
  STO6  h8b.sto  STO
  API6  h8b.api  VICAPI
  OC6   h8b.oc   OC
END PACKAGES
"""
)


(MF6_DIR / "h8b.api").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND {NODES}
END DIMENSIONS
"""
)


(MF6_DIR / "h8b.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h8b.hds
  BUDGET FILEOUT h8b.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
)


# ============================================================
# H8a topology reconstruction
#
# IMPORTANT:
# This is only to interpret positions in FLOWJA.
# The actual MODFLOW geometry/connectivity is being read from
# the validated copied H8a DISU input.
#
# H8a topology:
#
#     0 -- 1 -- 2 -- 3
#     |    |    |    |
#     4 -- 5 -- 6 -- 7
#     |    |    |    |
#     8 -- 9 --10 --11
#
# H8a wrote:
#
#     JA = [self, sorted(neighbors)]
#
# for each cell.
# ============================================================

def node(
    r,
    c,
):

    return (
        r *
        NCOL +
        c
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
        f"Expected H8a NJA=46; "
        f"reconstructed {NJA}."
    )


if ja.size != NJA:

    raise RuntimeError(
        "JA reconstruction size mismatch."
    )


# Starting FLOWJA position for each cell.
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


print()
print("===== CONNECTED TOPOLOGY =====")

print(
    "NODES =",
    NODES
)

print(
    "NJA   =",
    NJA
)

print(
    "IAC   =",
    iac
)


# ============================================================
# Storage coefficients
# ============================================================

storage_coeff_m2 = (
    SS_PER_M *
    THICKNESS_M *
    area_m2
)


# ============================================================
# Run connected MF6 with H7d VIC interface stress
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
            "H8B"
        )
    )


    flowja = mf6.get_value_ptr(
        mf6.get_var_address(
            "FLOWJA",
            "H8B"
        )
    )


    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H8B",
            "VICAPI"
        )
    )


    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H8B",
            "VICAPI"
        )
    )


    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H8B",
            "VICAPI"
        )
    )


    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H8B",
            "VICAPI"
        )
    )


    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H8B",
            "VICAPI"
        )
    )


    print()
    print("===== XMI ARRAYS =====")

    print(
        "X       =",
        head.shape
    )

    print(
        "FLOWJA  =",
        flowja.shape
    )

    print(
        "RHS     =",
        rhs.shape
    )

    print(
        "SIMVALS =",
        simvals.shape
    )


    if flowja.size != NJA:

        raise RuntimeError(
            f"FLOWJA size {flowja.size} "
            f"does not equal NJA {NJA}."
        )


    head_before = np.array(
        head,
        dtype=float,
        copy=True,
    )


    if not np.allclose(
        head_before,
        initial_head_m,
        rtol=0.0,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "Connected MF6 initial heads differ "
            "from H7d initial heads."
        )


    mf6.prepare_time_step(
        DT_DAYS
    )


    nbound[0] = NODES

    nodelist[:NODES] = np.arange(
        1,
        NODES + 1,
        dtype=nodelist.dtype,
    )


    # H2 sign convention:
    #
    # Q > 0 = water into MF6
    #
    # API:
    #
    # Q = HCOF*h - RHS
    #
    # pure specified flux:
    #
    # HCOF = 0
    # RHS  = -Q

    hcof[:NODES] = 0.0

    rhs[:NODES] = (
        -interface_volume_m3 /
        DT_DAYS
    )


    component = 1

    mf6.prepare_solve(
        component
    )


    converged = False
    iterations = 0


    for it in range(
        1,
        101
    ):

        nbound[0] = NODES

        nodelist[:NODES] = np.arange(
            1,
            NODES + 1,
            dtype=nodelist.dtype,
        )

        hcof[:NODES] = 0.0

        rhs[:NODES] = (
            -interface_volume_m3 /
            DT_DAYS
        )


        converged = mf6.solve(
            component
        )

        iterations = it


        if converged:
            break


    if not converged:

        raise RuntimeError(
            "H8b MF6 solve did not converge."
        )


    # xmi_solve() obtains the numerical solution, but MODFLOW 6
    # computes package flows and budgets during finalize_solve().
    #
    # Therefore FLOWJA and API SIMVALS must be sampled AFTER
    # finalize_solve(), not before it.

    mf6.finalize_solve(
        component
    )


    head_after = np.array(
        head,
        dtype=float,
        copy=True,
    )


    flowja_after = np.array(
        flowja,
        dtype=float,
        copy=True,
    )


    api_flow = np.array(
        simvals[:NODES],
        dtype=float,
        copy=True,
    )


    mf6.finalize_time_step()


finally:

    mf6.finalize()


# ============================================================
# Decode FLOWJA
#
# Skip the first (self/diagonal) JA entry for each cell.
#
# Positive off-diagonal FLOWJA is interpreted as flow INTO
# the current cell.
# ============================================================

lateral_rate_m3_day = np.zeros(
    NODES,
    dtype=float,
)


pair_rows = []

pair_antisymmetry = []


# Store directed flows.
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


    # First position is self.
    if ja[start] != n:

        raise RuntimeError(
            f"Cell {n} JA does not begin "
            "with itself."
        )


    for pos in range(
        start + 1,
        end
    ):

        m = int(
            ja[pos]
        )


        q = float(
            flowja_after[pos]
        )


        directed[
            (
                n,
                m
            )
        ] = q


        lateral_rate_m3_day[n] += (
            q
        )


# ============================================================
# Pairwise antisymmetry
#
# Q(n<-m) should equal -Q(m<-n).
# ============================================================

seen = set()


for (
    n,
    m
), q_nm in directed.items():

    key = tuple(
        sorted(
            (
                n,
                m
            )
        )
    )


    if key in seen:
        continue


    seen.add(
        key
    )


    q_mn = directed[
        (
            m,
            n
        )
    ]


    anti = (
        q_nm +
        q_mn
    )


    pair_antisymmetry.append(
        anti
    )


    pair_rows.append({
        "cell_a":
            key[0],

        "cell_b":
            key[1],

        "flow_into_a_from_b_m3_day":
            directed[
                (
                    key[0],
                    key[1]
                )
            ],

        "flow_into_b_from_a_m3_day":
            directed[
                (
                    key[1],
                    key[0]
                )
            ],

        "antisymmetry_error_m3_day":
            anti,
    })


max_pair_antisymmetry = float(
    np.max(
        np.abs(
            pair_antisymmetry
        )
    )
)


lateral_volume_m3 = (
    lateral_rate_m3_day *
    DT_DAYS
)


net_lateral_volume_m3 = float(
    lateral_volume_m3.sum()
)


# ============================================================
# Storage response
# ============================================================

head_change_m = (
    head_after -
    head_before
)


storage_change_m3 = (
    head_change_m *
    storage_coeff_m2
)


# Fundamental H8b equation:
#
# storage
#     =
# VIC interface
#     +
# net lateral inflow
#
cell_budget_residual_m3 = (
    storage_change_m3 -
    interface_volume_m3 -
    lateral_volume_m3
)


# ============================================================
# API check
# ============================================================

api_error_m3_day = (
    api_flow -
    interface_volume_m3 /
    DT_DAYS
)


# ============================================================
# Domain balances
# ============================================================

total_storage_m3 = float(
    storage_change_m3.sum()
)


domain_interface_error_m3 = (
    total_storage_m3 -
    interface_total_m3
)


full_budget_error_m3 = (
    total_storage_m3 -
    interface_total_m3 -
    net_lateral_volume_m3
)


# ============================================================
# Compare with H7d disconnected solution
#
# Same starting heads and same VIC boundary exchange.
#
# Difference in final spatial head field must therefore arise
# solely from lateral groundwater redistribution.
# ============================================================

connected_minus_disconnected_head_m = (
    head_after -
    h7d_disconnected_final_head_m
)


max_lateral_head_effect_m = float(
    np.max(
        np.abs(
            connected_minus_disconnected_head_m
        )
    )
)


max_lateral_volume_m3 = float(
    np.max(
        np.abs(
            lateral_volume_m3
        )
    )
)


max_cell_budget_residual_m3 = float(
    np.max(
        np.abs(
            cell_budget_residual_m3
        )
    )
)


max_api_error_m3_day = float(
    np.max(
        np.abs(
            api_error_m3_day
        )
    )
)


# ============================================================
# Save cell diagnostics
# ============================================================

cell_rows = []


for n in range(
    NODES
):

    cell_rows.append({
        "mf6_id":
            n,

        "area_m2":
            area_m2[n],

        "head_before_m":
            head_before[n],

        "interface_volume_m3":
            interface_volume_m3[n],

        "lateral_net_rate_m3_day":
            lateral_rate_m3_day[n],

        "lateral_net_volume_m3":
            lateral_volume_m3[n],

        "head_after_connected_m":
            head_after[n],

        "head_after_disconnected_h7d_m":
            h7d_disconnected_final_head_m[n],

        "connected_minus_disconnected_head_m":
            connected_minus_disconnected_head_m[n],

        "storage_change_m3":
            storage_change_m3[n],

        "expected_storage_from_interface_plus_lateral_m3":
            (
                interface_volume_m3[n] +
                lateral_volume_m3[n]
            ),

        "cell_budget_residual_m3":
            cell_budget_residual_m3[n],

        "api_simvals_m3_day":
            api_flow[n],

        "api_error_m3_day":
            api_error_m3_day[n],
    })


with (
    OUT /
    "cell_budget.csv"
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


with (
    OUT /
    "lateral_pairs.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            pair_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        pair_rows
    )


# ============================================================
# Pass/fail
# ============================================================

FLOWJA_SIZE_PASS = (
    flowja_after.size ==
    NJA
)


PAIRWISE_PASS = (
    max_pair_antisymmetry <
    1.0e-6
)


LATERAL_DOMAIN_PASS = (
    abs(
        net_lateral_volume_m3
    ) <
    1.0e-6
)


API_PASS = (
    max_api_error_m3_day <
    1.0e-6
)


CELL_BUDGET_PASS = (
    max_cell_budget_residual_m3 <
    1.0e-2
)


DOMAIN_BUDGET_PASS = (
    abs(
        full_budget_error_m3
    ) <
    1.0e-3
)


INTERFACE_TOTAL_PASS = (
    abs(
        domain_interface_error_m3
    ) <
    1.0e-3
)


LATERAL_EFFECT_PASS = (
    max_lateral_head_effect_m >
    1.0e-8
    and
    max_lateral_volume_m3 >
    1.0e-6
)


OVERALL = all([
    FLOWJA_SIZE_PASS,
    PAIRWISE_PASS,
    LATERAL_DOMAIN_PASS,
    API_PASS,
    CELL_BUDGET_PASS,
    DOMAIN_BUDGET_PASS,
    INTERFACE_TOTAL_PASS,
    LATERAL_EFFECT_PASS,
])


# ============================================================
# Summary
# ============================================================

summary = {
    "nodes":
        NODES,

    "NJA":
        NJA,

    "mf6_iterations":
        iterations,

    "interface_total_m3":
        interface_total_m3,

    "storage_total_m3":
        total_storage_m3,

    "net_lateral_volume_m3":
        net_lateral_volume_m3,

    "domain_storage_minus_interface_m3":
        domain_interface_error_m3,

    "full_domain_budget_error_m3":
        full_budget_error_m3,

    "max_pairwise_flowja_antisymmetry_m3_day":
        max_pair_antisymmetry,

    "max_api_error_m3_day":
        max_api_error_m3_day,

    "max_cell_budget_residual_m3":
        max_cell_budget_residual_m3,

    "max_abs_lateral_volume_m3":
        max_lateral_volume_m3,

    "max_connected_minus_disconnected_head_m":
        max_lateral_head_effect_m,

    "flowja_size_pass":
        FLOWJA_SIZE_PASS,

    "pairwise_flow_pass":
        PAIRWISE_PASS,

    "lateral_domain_conservation_pass":
        LATERAL_DOMAIN_PASS,

    "api_pass":
        API_PASS,

    "cell_budget_pass":
        CELL_BUDGET_PASS,

    "domain_budget_pass":
        DOMAIN_BUDGET_PASS,

    "interface_total_pass":
        INTERFACE_TOTAL_PASS,

    "lateral_effect_pass":
        LATERAL_EFFECT_PASS,

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
# Print
# ============================================================

print()
print("=" * 132)
print("H8b CONNECTED MF6 + VIC INTERFACE CELL BUDGET")
print("=" * 132)

print(
    f"{'id':>3s} "
    f"{'V interface':>14s} "
    f"{'V lateral':>14s} "
    f"{'V storage':>14s} "
    f"{'budget err':>12s} "
    f"{'h conn':>13s} "
    f"{'h disconn':>13s} "
    f"{'dh lateral':>12s}"
)


for r in cell_rows:

    print(
        f"{r['mf6_id']:3d} "
        f"{r['interface_volume_m3']:14.3f} "
        f"{r['lateral_net_volume_m3']:14.3f} "
        f"{r['storage_change_m3']:14.3f} "
        f"{r['cell_budget_residual_m3']:12.3e} "
        f"{r['head_after_connected_m']:13.7f} "
        f"{r['head_after_disconnected_h7d_m']:13.7f} "
        f"{r['connected_minus_disconnected_head_m']:12.5e}"
    )


print()
print("=" * 88)
print("H8b VALIDATION")
print("=" * 88)

print(
    "FLOWJA SIZE:",
    "PASS"
    if FLOWJA_SIZE_PASS
    else "FAIL",
    f"({flowja_after.size}/{NJA})"
)

print(
    "PAIRWISE LATERAL ANTISYMMETRY:",
    "PASS"
    if PAIRWISE_PASS
    else "FAIL",
    "max error =",
    max_pair_antisymmetry,
    "m3/day"
)

print(
    "NET LATERAL DOMAIN FLOW:",
    "PASS"
    if LATERAL_DOMAIN_PASS
    else "FAIL",
    "net =",
    net_lateral_volume_m3,
    "m3"
)

print(
    "API INTERFACE FLOWS:",
    "PASS"
    if API_PASS
    else "FAIL",
    "max error =",
    max_api_error_m3_day,
    "m3/day"
)

print(
    "PER-CELL COMBINED BUDGET:",
    "PASS"
    if CELL_BUDGET_PASS
    else "FAIL",
    "max residual =",
    max_cell_budget_residual_m3,
    "m3"
)

print(
    "DOMAIN STORAGE = VIC INTERFACE:",
    "PASS"
    if INTERFACE_TOTAL_PASS
    else "FAIL",
    "error =",
    domain_interface_error_m3,
    "m3"
)

print(
    "FULL DOMAIN BUDGET:",
    "PASS"
    if DOMAIN_BUDGET_PASS
    else "FAIL",
    "error =",
    full_budget_error_m3,
    "m3"
)

print(
    "LATERAL FLOW CHANGES HEAD FIELD:",
    "PASS"
    if LATERAL_EFFECT_PASS
    else "FAIL"
)


print()
print("===== DOMAIN TOTALS =====")

print(
    "VIC/interface volume =",
    interface_total_m3,
    "m3"
)

print(
    "net lateral volume   =",
    net_lateral_volume_m3,
    "m3"
)

print(
    "MF6 storage change   =",
    total_storage_m3,
    "m3"
)

print(
    "full budget error    =",
    full_budget_error_m3,
    "m3"
)


print()
print("===== LATERAL EFFECT =====")

print(
    "max |lateral cell volume| =",
    max_lateral_volume_m3,
    "m3"
)

print(
    "max |connected - H7d head| =",
    max_lateral_head_effect_m,
    "m"
)


print()
print("=" * 88)

print(
    "H8b CONNECTED COUPLED CELL BUDGET:",
    "PASS"
    if OVERALL
    else "REQUIRES REVIEW"
)

print("=" * 88)


print()
print("Saved:")
print(OUT / "cell_budget.csv")
print(OUT / "lateral_pairs.csv")
print(OUT / "summary.csv")

print()
print("Experiment directory:")
print(OUT)
