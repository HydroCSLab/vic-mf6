#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv
import time

import numpy as np
from pyproj import Geod
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

H7B = (
    ROOT /
    "H7b_stehekin_real_mapper"
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    ROOT /
    f"H8a_connected_mf6_lateral_{STAMP}"
)

MF6_DIR = OUT / "mf6"

MF6_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Model controls
# ============================================================

NROW = 3
NCOL = 4
NODES = NROW * NCOL

NDAYS = 10

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)

SS_PER_M = 1.0e-5

# Numerical connectivity test only.
K_M_DAY = 50.0


# ============================================================
# Read H7b MF6 cells
# ============================================================

with (
    H7B /
    "mf6_cells.csv"
).open() as f:

    rows = list(
        csv.DictReader(f)
    )


rows.sort(
    key=lambda x:
        int(x["mf6_id"])
)


if len(rows) != NODES:

    raise RuntimeError(
        f"Expected {NODES} MF6 cells; "
        f"found {len(rows)}"
    )


area_m2 = np.array(
    [
        float(
            x["rectangle_area_m2"]
        )
        for x in rows
    ],
    dtype=float,
)


initial_head_m = np.array(
    [
        float(
            x["head_m"]
        )
        for x in rows
    ],
    dtype=float,
)


cell_row = np.array(
    [
        int(x["row"])
        for x in rows
    ],
    dtype=int,
)


cell_col = np.array(
    [
        int(x["col"])
        for x in rows
    ],
    dtype=int,
)


center_lon = np.array(
    [
        float(x["center_lon"])
        for x in rows
    ],
    dtype=float,
)


center_lat = np.array(
    [
        float(x["center_lat"])
        for x in rows
    ],
    dtype=float,
)


if not np.array_equal(
    np.arange(NODES),
    np.array(
        [
            int(x["mf6_id"])
            for x in rows
        ]
    ),
):

    raise RuntimeError(
        "MF6 cells are not numbered 0..11."
    )


# ============================================================
# Recover conceptual grid boundaries
#
# H7b cells are rectangles.  Recover boundary coordinates from
# the stored cell centers and the known outer VIC active-domain
# envelope.
# ============================================================

with (
    H7B /
    "vic_cells.csv"
).open() as f:

    vr = list(
        csv.DictReader(f)
    )


vic_lon = np.array(
    [
        float(x["lon"])
        for x in vr
    ]
)

vic_lat = np.array(
    [
        float(x["lat"])
        for x in vr
    ]
)


ulon = np.unique(
    np.sort(
        vic_lon
    )
)

ulat = np.unique(
    np.sort(
        vic_lat
    )
)


DX_DEG = float(
    np.median(
        np.diff(
            ulon
        )
    )
)

DY_DEG = float(
    np.median(
        np.diff(
            ulat
        )
    )
)


WEST = float(
    vic_lon.min() -
    0.5 * DX_DEG
)

EAST = float(
    vic_lon.max() +
    0.5 * DX_DEG
)

SOUTH = float(
    vic_lat.min() -
    0.5 * DY_DEG
)

NORTH = float(
    vic_lat.max() +
    0.5 * DY_DEG
)


xcenter = np.array(
    [
        center_lon[
            np.where(
                cell_row == 0
            )[0][
                np.where(
                    cell_col[
                        cell_row == 0
                    ] == c
                )[0][0]
            ]
        ]
        for c in range(NCOL)
    ]
)


ycenter = np.array(
    [
        center_lat[
            np.where(
                cell_col == 0
            )[0][
                np.where(
                    cell_row[
                        cell_col == 0
                    ] == r
                )[0][0]
            ]
        ]
        for r in range(NROW)
    ]
)


xbounds = np.empty(
    NCOL + 1,
    dtype=float,
)

ybounds = np.empty(
    NROW + 1,
    dtype=float,
)


xbounds[0] = WEST

for c in range(NCOL):

    xbounds[c + 1] = (
        2.0 *
        xcenter[c] -
        xbounds[c]
    )


ybounds[0] = SOUTH

for r in range(NROW):

    ybounds[r + 1] = (
        2.0 *
        ycenter[r] -
        ybounds[r]
    )


if not np.isclose(
    xbounds[-1],
    EAST,
    atol=1.0e-10,
):

    raise RuntimeError(
        "Could not reconstruct MF6 x boundaries."
    )


if not np.isclose(
    ybounds[-1],
    NORTH,
    atol=1.0e-10,
):

    raise RuntimeError(
        "Could not reconstruct MF6 y boundaries."
    )


# ============================================================
# Metric connection geometry
#
# Convert longitude/latitude spans to representative metric
# widths/heights with geodesic distances.
# ============================================================

GEOD = Geod(
    ellps="GRS80"
)

mid_lat = (
    0.5 *
    (
        SOUTH +
        NORTH
    )
)

mid_lon = (
    0.5 *
    (
        WEST +
        EAST
    )
)


width_m = np.empty(
    NCOL,
    dtype=float,
)

height_m = np.empty(
    NROW,
    dtype=float,
)


for c in range(NCOL):

    _, _, dist = GEOD.inv(
        xbounds[c],
        mid_lat,
        xbounds[c + 1],
        mid_lat,
    )

    width_m[c] = abs(
        dist
    )


for r in range(NROW):

    _, _, dist = GEOD.inv(
        mid_lon,
        ybounds[r],
        mid_lon,
        ybounds[r + 1],
    )

    height_m[r] = abs(
        dist
    )


print()
print("===== H8a GRID =====")

print(
    "x bounds:",
    xbounds
)

print(
    "y bounds:",
    ybounds
)

print(
    "column widths [m]:",
    width_m
)

print(
    "row heights [m]:",
    height_m
)

print(
    "full area [m2] =",
    area_m2.sum()
)

print(
    "K =",
    K_M_DAY,
    "m/day"
)

print(
    "Ss =",
    SS_PER_M,
    "1/m"
)


# ============================================================
# DISU topology
#
# Orthogonal 3 x 4 connectivity:
#
#   0--1--2--3
#   |  |  |  |
#   4--5--6--7
#   |  |  |  |
#   8--9-10--11
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
    for n in range(NODES)
}


for r in range(NROW):

    for c in range(NCOL):

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


iac = []

ja = []

ihc = []

cl12 = []

hwva = []


for n in range(NODES):

    r = int(
        cell_row[n]
    )

    c = int(
        cell_col[n]
    )

    connection_list = [
        n
    ] + neighbors[n]


    iac.append(
        len(
            connection_list
        )
    )


    for m in connection_list:

        # MODFLOW node numbering is 1-based.
        ja.append(
            m + 1
        )


        # Diagonal/self entry.
        if m == n:

            ihc.append(
                0
            )

            cl12.append(
                0.0
            )

            hwva.append(
                0.0
            )

            continue


        mr = int(
            cell_row[m]
        )

        mc = int(
            cell_col[m]
        )


        # Horizontal x-direction neighbor.
        if mr == r:

            ihc.append(
                1
            )

            # Distance from THIS cell center to face.
            cl12.append(
                0.5 *
                width_m[c]
            )

            # Face width perpendicular to horizontal flow.
            hwva.append(
                height_m[r]
            )


        # Horizontal y-direction neighbor.
        elif mc == c:

            ihc.append(
                1
            )

            cl12.append(
                0.5 *
                height_m[r]
            )

            hwva.append(
                width_m[c]
            )


        else:

            raise RuntimeError(
                "Unexpected diagonal connection."
            )


iac = np.asarray(
    iac,
    dtype=int,
)

ja = np.asarray(
    ja,
    dtype=int,
)

ihc = np.asarray(
    ihc,
    dtype=int,
)

cl12 = np.asarray(
    cl12,
    dtype=float,
)

hwva = np.asarray(
    hwva,
    dtype=float,
)


NJA = int(
    iac.sum()
)


if (
    len(ja) != NJA or
    len(ihc) != NJA or
    len(cl12) != NJA or
    len(hwva) != NJA
):

    raise RuntimeError(
        "DISU connection-array size mismatch."
    )


# Check symmetric HWVA explicitly.
position = 0

hwva_lookup = {}


for n in range(NODES):

    for k in range(
        iac[n]
    ):

        m = (
            ja[position] -
            1
        )

        if m != n:

            hwva_lookup[
                (
                    n,
                    m
                )
            ] = hwva[
                position
            ]

        position += 1


for (
    n,
    m
), value in hwva_lookup.items():

    reverse = hwva_lookup[
        (
            m,
            n
        )
    ]

    if not np.isclose(
        value,
        reverse,
        rtol=0.0,
        atol=1.0e-10,
    ):

        raise RuntimeError(
            f"HWVA not symmetric for "
            f"{n}<->{m}"
        )


print()
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
# Write MF6 input
# ============================================================

def array_text(
    a,
):

    return " ".join(
        f"{x:.17g}"
        if np.issubdtype(
            np.asarray(a).dtype,
            np.floating
        )
        else str(int(x))
        for x in a
    )


(MF6_DIR / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h8a.tdis
END TIMING

BEGIN MODELS
  GWF6 h8a.nam H8A
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h8a.ims H8A
END SOLUTIONGROUP
"""
)


(MF6_DIR / "h8a.tdis").write_text(
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


(MF6_DIR / "h8a.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY MODERATE
END OPTIONS
"""
)


(MF6_DIR / "h8a.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DISU6 h8a.disu DISU
  IC6   h8a.ic   IC
  NPF6  h8a.npf  NPF
  STO6  h8a.sto  STO
  OC6   h8a.oc   OC
END PACKAGES
"""
)


(MF6_DIR / "h8a.disu").write_text(
f"""BEGIN OPTIONS
  LENGTH_UNITS METERS
END OPTIONS

BEGIN DIMENSIONS
  NODES {NODES}
  NJA {NJA}
END DIMENSIONS

BEGIN GRIDDATA
  TOP
    CONSTANT {TOP_M:.17g}

  BOT
    CONSTANT {BOT_M:.17g}

  AREA
    INTERNAL FACTOR 1.0
    {array_text(area_m2)}
END GRIDDATA

BEGIN CONNECTIONDATA
  IAC
    INTERNAL FACTOR 1
    {array_text(iac)}

  JA
    INTERNAL FACTOR 1
    {array_text(ja)}

  IHC
    INTERNAL FACTOR 1
    {array_text(ihc)}

  CL12
    INTERNAL FACTOR 1.0
    {array_text(cl12)}

  HWVA
    INTERNAL FACTOR 1.0
    {array_text(hwva)}
END CONNECTIONDATA
"""
)


(MF6_DIR / "h8a.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    INTERNAL FACTOR 1.0
    {array_text(initial_head_m)}
END GRIDDATA
"""
)


(MF6_DIR / "h8a.npf").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICELLTYPE
    CONSTANT 0

  K
    CONSTANT {K_M_DAY:.17g}
END GRIDDATA
"""
)


(MF6_DIR / "h8a.sto").write_text(
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


(MF6_DIR / "h8a.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h8a.hds
  BUDGET FILEOUT h8a.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
)


# ============================================================
# Storage invariants
# ============================================================

storage_coeff_m2 = (
    SS_PER_M *
    THICKNESS_M *
    area_m2
)


def weighted_mean_head(
    h,
):

    return float(
        np.sum(
            h *
            storage_coeff_m2
        ) /
        np.sum(
            storage_coeff_m2
        )
    )


initial_weighted_head = (
    weighted_mean_head(
        initial_head_m
    )
)


# ============================================================
# XMI solve
# ============================================================

mf6 = XmiWrapper(
    str(LIBMF6),
    working_directory=
        str(MF6_DIR),
)


mf6.initialize_mpi(MPI.COMM_SELF.py2f())


daily = []

cell_daily = []


try:

    head = mf6.get_value_ptr(
        mf6.get_var_address(
            "X",
            "H8A"
        )
    )


    h0 = np.array(
        head,
        dtype=float,
        copy=True,
    )


    if not np.allclose(
        h0,
        initial_head_m,
        rtol=0.0,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "Initial MF6 head mismatch."
        )


    previous_min = float(
        h0.min()
    )

    previous_max = float(
        h0.max()
    )


    for day in range(
        1,
        NDAYS + 1
    ):

        h_before = np.array(
            head,
            dtype=float,
            copy=True,
        )


        mf6.prepare_time_step(
            1.0
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

            converged = mf6.solve(
                component
            )

            iterations = it


            if converged:
                break


        if not converged:

            raise RuntimeError(
                f"MF6 failed on day {day}."
            )


        mf6.finalize_solve(
            component
        )

        mf6.finalize_time_step()


        h_after = np.array(
            head,
            dtype=float,
            copy=True,
        )


        daily_storage_m3 = (
            (
                h_after -
                h_before
            ) *
            storage_coeff_m2
        )


        cumulative_storage_m3 = (
            (
                h_after -
                h0
            ) *
            storage_coeff_m2
        )


        daily_net_storage = float(
            daily_storage_m3.sum()
        )


        cumulative_net_storage = float(
            cumulative_storage_m3.sum()
        )


        weighted_head = (
            weighted_mean_head(
                h_after
            )
        )


        weighted_head_error = (
            weighted_head -
            initial_weighted_head
        )


        hmin = float(
            h_after.min()
        )

        hmax = float(
            h_after.max()
        )

        hrange = (
            hmax -
            hmin
        )


        extrema_contract = (
            hmin >=
            previous_min - 1.0e-10
            and
            hmax <=
            previous_max + 1.0e-10
        )


        rising = int(
            np.sum(
                h_after >
                h_before
            )
        )

        falling = int(
            np.sum(
                h_after <
                h_before
            )
        )


        daily.append({
            "day":
                day,

            "mf6_iterations":
                iterations,

            "head_min_m":
                hmin,

            "head_max_m":
                hmax,

            "head_range_m":
                hrange,

            "weighted_mean_head_m":
                weighted_head,

            "weighted_mean_head_error_m":
                weighted_head_error,

            "daily_net_storage_change_m3":
                daily_net_storage,

            "cumulative_net_storage_change_m3":
                cumulative_net_storage,

            "rising_cells":
                rising,

            "falling_cells":
                falling,

            "extrema_contract":
                extrema_contract,
        })


        for n in range(
            NODES
        ):

            cell_daily.append({
                "day":
                    day,

                "mf6_id":
                    n,

                "row":
                    int(
                        cell_row[n]
                    ),

                "col":
                    int(
                        cell_col[n]
                    ),

                "area_m2":
                    area_m2[n],

                "head_start_m":
                    h_before[n],

                "head_end_m":
                    h_after[n],

                "head_change_m":
                    (
                        h_after[n] -
                        h_before[n]
                    ),

                "daily_storage_change_m3":
                    daily_storage_m3[n],

                "cumulative_storage_change_m3":
                    cumulative_storage_m3[n],
            })


        print(
            f"day {day:2d}: "
            f"h=[{hmin:.9f}, {hmax:.9f}] "
            f"range={hrange:.9f} "
            f"rise/fall={rising}/{falling} "
            f"dV={daily_net_storage:.3e} m3 "
            f"dHmean={weighted_head_error:.3e} m"
        )


        previous_min = (
            hmin
        )

        previous_max = (
            hmax
        )


    final_head_m = np.array(
        head,
        dtype=float,
        copy=True,
    )


finally:

    mf6.finalize()


# ============================================================
# Validation
# ============================================================

initial_range = float(
    h0.max() -
    h0.min()
)

final_range = float(
    final_head_m.max() -
    final_head_m.min()
)


range_reduction = (
    initial_range -
    final_range
)


max_daily_storage_error = max(
    abs(
        x[
            "daily_net_storage_change_m3"
        ]
    )
    for x in daily
)


final_storage_error = abs(
    daily[-1][
        "cumulative_net_storage_change_m3"
    ]
)


max_weighted_head_error = max(
    abs(
        x[
            "weighted_mean_head_error_m"
        ]
    )
    for x in daily
)


all_extrema_contract = all(
    x[
        "extrema_contract"
    ]
    for x in daily
)


redistribution_pass = (
    final_range <
    initial_range
)


daily_conservation_pass = (
    max_daily_storage_error <
    1.0e-3
)


cumulative_conservation_pass = (
    final_storage_error <
    1.0e-3
)


weighted_mean_pass = (
    max_weighted_head_error <
    1.0e-10
)


extrema_pass = (
    all_extrema_contract
)


overall = all([
    redistribution_pass,
    daily_conservation_pass,
    cumulative_conservation_pass,
    weighted_mean_pass,
    extrema_pass,
])


# ============================================================
# Save
# ============================================================

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
            daily[0].keys()
    )

    w.writeheader()
    w.writerows(
        daily
    )


with (
    OUT /
    "cell_daily.csv"
).open(
    "w",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=
            cell_daily[0].keys()
    )

    w.writeheader()
    w.writerows(
        cell_daily
    )


summary = {
    "nodes":
        NODES,

    "NJA":
        NJA,

    "days":
        NDAYS,

    "K_m_day":
        K_M_DAY,

    "Ss_per_m":
        SS_PER_M,

    "initial_head_min_m":
        float(
            h0.min()
        ),

    "initial_head_max_m":
        float(
            h0.max()
        ),

    "initial_head_range_m":
        initial_range,

    "final_head_min_m":
        float(
            final_head_m.min()
        ),

    "final_head_max_m":
        float(
            final_head_m.max()
        ),

    "final_head_range_m":
        final_range,

    "head_range_reduction_m":
        range_reduction,

    "initial_storage_weighted_head_m":
        initial_weighted_head,

    "final_storage_weighted_head_m":
        weighted_mean_head(
            final_head_m
        ),

    "max_weighted_head_error_m":
        max_weighted_head_error,

    "max_daily_net_storage_error_m3":
        max_daily_storage_error,

    "final_cumulative_storage_error_m3":
        final_storage_error,

    "redistribution_pass":
        redistribution_pass,

    "extrema_contraction_pass":
        extrema_pass,

    "daily_conservation_pass":
        daily_conservation_pass,

    "cumulative_conservation_pass":
        cumulative_conservation_pass,

    "weighted_mean_head_pass":
        weighted_mean_pass,

    "overall_pass":
        overall,
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


# ============================================================
# Print
# ============================================================

print()
print("=" * 88)
print("H8a CONNECTED MF6 LATERAL-REDISTRIBUTION TEST")
print("=" * 88)

print(
    "initial head range       =",
    initial_range,
    "m"
)

print(
    "final head range         =",
    final_range,
    "m"
)

print(
    "range reduction          =",
    range_reduction,
    "m"
)

print()
print(
    "initial weighted head    =",
    initial_weighted_head,
    "m"
)

print(
    "final weighted head      =",
    weighted_mean_head(
        final_head_m
    ),
    "m"
)

print(
    "max weighted-head error =",
    max_weighted_head_error,
    "m"
)

print()
print(
    "max daily net storage    =",
    max_daily_storage_error,
    "m3"
)

print(
    "final net storage error  =",
    final_storage_error,
    "m3"
)

print()
print(
    "HEAD REDISTRIBUTION:",
    "PASS"
    if redistribution_pass
    else "FAIL"
)

print(
    "EXTREMA CONTRACTION:",
    "PASS"
    if extrema_pass
    else "FAIL"
)

print(
    "DAILY STORAGE CONSERVATION:",
    "PASS"
    if daily_conservation_pass
    else "FAIL"
)

print(
    "CUMULATIVE STORAGE CONSERVATION:",
    "PASS"
    if cumulative_conservation_pass
    else "FAIL"
)

print(
    "STORAGE-WEIGHTED MEAN HEAD:",
    "PASS"
    if weighted_mean_pass
    else "FAIL"
)


print()
print("=" * 88)

print(
    "H8a CONNECTED MF6:",
    "PASS"
    if overall
    else "REQUIRES REVIEW"
)

print("=" * 88)

print()
print("Saved:")
print(OUT / "daily.csv")
print(OUT / "cell_daily.csv")
print(OUT / "summary.csv")

print()
print("Experiment directory:")
print(OUT)
