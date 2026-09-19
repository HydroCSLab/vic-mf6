#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv

import netCDF4 as nc
import numpy as np

try:
    from pyproj import Geod
except ImportError as exc:
    raise RuntimeError(
        "H7b requires pyproj for geodetic overlap areas."
    ) from exc


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    str(RESULT_ROOT)
)

DOMAIN = Path(
    str(SAMPLE_ROOT / 'parameters/domain.stehekin.20151028.nc')
)

OUT = (
    ROOT /
    "H7b_stehekin_real_mapper"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# Prefer the already validated H1 fixed-head run.
q_candidates = [
    ROOT /
    "H1_mf6_to_vic_head" /
    "fluxes.mf6_head.nc",
]

q_candidates.extend(
    sorted(
        ROOT.glob(
            "**/fluxes.mf6_head.nc"
        )
    )
)

QFILE = next(
    (
        p
        for p in q_candidates
        if p.exists()
    ),
    None
)

if QFILE is None:
    raise RuntimeError(
        "Could not find the validated "
        "H1 fluxes.mf6_head.nc file."
    )


print()
print("===== H7b INPUT =====")
print("domain =", DOMAIN)
print("flux   =", QFILE)


# ============================================================
# Helpers
# ============================================================

def get_variable(
    ds,
    names,
):

    for name in names:

        if name in ds.variables:
            return np.asarray(
                ds[name][:]
            )

    raise RuntimeError(
        "Could not find any variable from: "
        + ", ".join(names)
    )


def coordinate_vectors(
    lat,
    lon,
):

    # Standard VIC rectilinear coordinate case.
    if (
        lat.ndim == 1 and
        lon.ndim == 1
    ):

        return (
            lat.astype(float),
            lon.astype(float),
        )


    # Also support 2-D rectilinear coordinates.
    if (
        lat.ndim == 2 and
        lon.ndim == 2
    ):

        latvec = lat[:, 0]
        lonvec = lon[0, :]


        if not np.allclose(
            lat,
            latvec[:, None],
            rtol=0.0,
            atol=1.0e-12,
        ):

            raise RuntimeError(
                "Latitude grid is not rectilinear."
            )


        if not np.allclose(
            lon,
            lonvec[None, :],
            rtol=0.0,
            atol=1.0e-12,
        ):

            raise RuntimeError(
                "Longitude grid is not rectilinear."
            )


        return (
            latvec.astype(float),
            lonvec.astype(float),
        )


    raise RuntimeError(
        "Unsupported lat/lon coordinate geometry."
    )


def edges_from_centers(
    centers,
):

    centers = np.asarray(
        centers,
        dtype=float,
    )


    if centers.size < 2:
        raise RuntimeError(
            "Need at least two coordinate centers."
        )


    edges = np.empty(
        centers.size + 1,
        dtype=float,
    )


    edges[1:-1] = (
        0.5 *
        (
            centers[:-1] +
            centers[1:]
        )
    )


    edges[0] = (
        centers[0] -
        0.5 *
        (
            centers[1] -
            centers[0]
        )
    )


    edges[-1] = (
        centers[-1] +
        0.5 *
        (
            centers[-1] -
            centers[-2]
        )
    )


    return edges


GEOD = Geod(
    ellps="GRS80"
)


def rectangle_area_m2(
    west,
    south,
    east,
    north,
):

    if (
        east <= west or
        north <= south
    ):
        return 0.0


    lons = [
        west,
        east,
        east,
        west,
    ]

    lats = [
        south,
        south,
        north,
        north,
    ]


    area, _ = GEOD.polygon_area_perimeter(
        lons,
        lats,
    )


    return abs(
        float(area)
    )


def overlap_rectangle(
    a,
    b,
):

    west = max(
        a["west"],
        b["west"],
    )

    east = min(
        a["east"],
        b["east"],
    )

    south = max(
        a["south"],
        b["south"],
    )

    north = min(
        a["north"],
        b["north"],
    )


    if (
        east <= west or
        north <= south
    ):

        return None


    return {
        "west":
            west,

        "east":
            east,

        "south":
            south,

        "north":
            north,
    }


# ============================================================
# Read actual VIC domain
# ============================================================

with nc.Dataset(DOMAIN) as ds:

    lat_raw = get_variable(
        ds,
        [
            "lat",
            "latitude",
        ],
    )

    lon_raw = get_variable(
        ds,
        [
            "lon",
            "longitude",
        ],
    )

    area = np.ma.asarray(
        ds["area"][:],
        dtype=float,
    )

    mask = (
        np.asarray(
            ds["mask"][:]
        ) > 0
    )


latvec, lonvec = coordinate_vectors(
    lat_raw,
    lon_raw,
)


lat_edges = edges_from_centers(
    latvec
)

lon_edges = edges_from_centers(
    lonvec
)


if area.shape != mask.shape:

    raise RuntimeError(
        "area and mask shapes differ."
    )


if area.shape != (
    latvec.size,
    lonvec.size,
):

    raise RuntimeError(
        "Domain dimensions do not match "
        "lat/lon coordinates."
    )


# ============================================================
# Actual day-1 VIC groundwater exchange
# ============================================================

with nc.Dataset(QFILE) as ds:

    q_all = np.ma.asarray(
        ds[
            "OUT_GW_EXCHANGE"
        ][:]
    )


if q_all.ndim != 3:

    raise RuntimeError(
        "Expected OUT_GW_EXCHANGE(time,y,x); "
        f"got shape {q_all.shape}"
    )


q = np.ma.asarray(
    q_all[0],
    dtype=float,
)


if q.shape != mask.shape:

    raise RuntimeError(
        "Flux grid does not match domain grid."
    )


# ============================================================
# Build actual active VIC cell polygons
# ============================================================

vic = []


for r in range(
    mask.shape[0]
):

    for c in range(
        mask.shape[1]
    ):

        if not mask[r, c]:
            continue


        if np.ma.is_masked(
            area[r, c]
        ):
            continue


        if np.ma.is_masked(
            q[r, c]
        ):
            raise RuntimeError(
                f"Active VIC cell ({r},{c}) "
                "has masked GW exchange."
            )


        x0 = lon_edges[c]
        x1 = lon_edges[c + 1]

        y0 = lat_edges[r]
        y1 = lat_edges[r + 1]


        west = min(x0, x1)
        east = max(x0, x1)

        south = min(y0, y1)
        north = max(y0, y1)


        vic.append({
            "id":
                len(vic),

            "row":
                r,

            "col":
                c,

            "lon":
                float(
                    lonvec[c]
                ),

            "lat":
                float(
                    latvec[r]
                ),

            "west":
                west,

            "east":
                east,

            "south":
                south,

            "north":
                north,

            "vic_area_m2":
                float(
                    area[r, c]
                ),

            "q_mm":
                float(
                    q[r, c]
                ),
        })


if len(vic) != 16:

    raise RuntimeError(
        "Expected 16 active Stehekin VIC cells; "
        f"found {len(vic)}."
    )


print()
print(
    "active VIC cells =",
    len(vic)
)


# ============================================================
# Active VIC envelope
# ============================================================

WEST = min(
    x["west"]
    for x in vic
)

EAST = max(
    x["east"]
    for x in vic
)

SOUTH = min(
    x["south"]
    for x in vic
)

NORTH = max(
    x["north"]
    for x in vic
)


print(
    "active envelope   =",
    WEST,
    SOUTH,
    EAST,
    NORTH,
)


# ============================================================
# Deliberately mismatched conceptual MF6 grid
#
# 4 columns x 3 rows.
#
# Fractions are deliberately chosen to avoid VIC cell
# boundaries.
# ============================================================

xfrac = np.array([
    0.00,
    0.18,
    0.52,
    0.79,
    1.00,
])

yfrac = np.array([
    0.00,
    0.27,
    0.61,
    1.00,
])


mf6_x = (
    WEST +
    xfrac *
    (
        EAST -
        WEST
    )
)

mf6_y = (
    SOUTH +
    yfrac *
    (
        NORTH -
        SOUTH
    )
)


mf6 = []


for r in range(
    len(mf6_y) - 1
):

    for c in range(
        len(mf6_x) - 1
    ):

        west = mf6_x[c]
        east = mf6_x[c + 1]

        south = mf6_y[r]
        north = mf6_y[r + 1]


        mf6.append({
            "id":
                len(mf6),

            "row":
                r,

            "col":
                c,

            "west":
                west,

            "east":
                east,

            "south":
                south,

            "north":
                north,

            "center_lon":
                0.5 *
                (
                    west +
                    east
                ),

            "center_lat":
                0.5 *
                (
                    south +
                    north
                ),

            "rectangle_area_m2":
                rectangle_area_m2(
                    west,
                    south,
                    east,
                    north,
                ),
        })


NV = len(vic)
NM = len(mf6)


print(
    "conceptual MF6 cells =",
    NM
)


# ============================================================
# Build raw geodetic overlap matrix
# ============================================================

Araw = np.zeros(
    (
        NV,
        NM
    ),
    dtype=float,
)


for i, vc in enumerate(vic):

    for j, mc in enumerate(mf6):

        rect = overlap_rectangle(
            vc,
            mc,
        )

        if rect is None:
            continue


        Araw[i, j] = (
            rectangle_area_m2(
                rect["west"],
                rect["south"],
                rect["east"],
                rect["north"],
            )
        )


# ============================================================
# Normalize each VIC row to VIC's actual stored area.
#
# This is important:
#
#   sum_j A_ij == actual VIC NetCDF area_i
#
# exactly to floating point.
# ============================================================

A = np.zeros_like(
    Araw
)


raw_vic_area = Araw.sum(
    axis=1
)


for i in range(NV):

    if raw_vic_area[i] <= 0.0:

        raise RuntimeError(
            f"VIC cell {i} has no MF6 overlap."
        )


    A[i, :] = (
        Araw[i, :] *
        (
            vic[i][
                "vic_area_m2"
            ] /
            raw_vic_area[i]
        )
    )


vic_area = np.array([
    x[
        "vic_area_m2"
    ]
    for x in vic
])


mf6_coupled_area = A.sum(
    axis=0
)


coverage_error = (
    A.sum(axis=1) -
    vic_area
)


max_coverage_error = float(
    np.max(
        np.abs(
            coverage_error
        )
    )
)


print()
print("=" * 76)
print("H7b ACTIVE-DOMAIN COVERAGE")
print("=" * 76)

print(
    "max VIC coverage error =",
    max_coverage_error,
    "m2"
)

print(
    "total VIC area         =",
    vic_area.sum(),
    "m2"
)

print(
    "total overlap area     =",
    A.sum(),
    "m2"
)


if not np.allclose(
    A.sum(axis=1),
    vic_area,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "VIC active-domain coverage failed."
    )


print(
    "ACTIVE VIC COVERAGE: PASS"
)


# ============================================================
# Forward mapping:
# actual VIC q -> conceptual MF6 cell volumes
# ============================================================

q_vic_mm = np.array([
    x["q_mm"]
    for x in vic
])


vic_volume_m3 = (
    q_vic_mm *
    1.0e-3 *
    vic_area
)


mf6_net_volume_m3 = (
    1.0e-3 *
    A.T.dot(
        q_vic_mm
    )
)


vic_total = float(
    vic_volume_m3.sum()
)

mf6_total = float(
    mf6_net_volume_m3.sum()
)


net_error = (
    mf6_total -
    vic_total
)


print()
print("=" * 76)
print("ACTUAL VIC -> MF6 SIGNED VOLUME")
print("=" * 76)

print(
    "VIC q min/max =",
    q_vic_mm.min(),
    q_vic_mm.max(),
    "mm"
)

print(
    "positive VIC cells =",
    int(
        np.sum(
            q_vic_mm > 0.0
        )
    )
)

print(
    "negative VIC cells =",
    int(
        np.sum(
            q_vic_mm < 0.0
        )
    )
)

print(
    "VIC total volume =",
    vic_total,
    "m3"
)

print(
    "MF6 total volume =",
    mf6_total,
    "m3"
)

print(
    "mapping error     =",
    net_error,
    "m3"
)


if not np.isclose(
    mf6_total,
    vic_total,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "Actual VIC->MF6 net conservation failed."
    )


print(
    "VIC -> MF6 NET CONSERVATION: PASS"
)


# ============================================================
# Gross positive / negative components
# ============================================================

q_positive = np.maximum(
    q_vic_mm,
    0.0
)

q_negative = np.minimum(
    q_vic_mm,
    0.0
)


mf6_positive = (
    1.0e-3 *
    A.T.dot(
        q_positive
    )
)

mf6_negative = (
    1.0e-3 *
    A.T.dot(
        q_negative
    )
)


vic_positive = float(
    np.sum(
        q_positive *
        1.0e-3 *
        vic_area
    )
)

vic_negative = float(
    np.sum(
        q_negative *
        1.0e-3 *
        vic_area
    )
)


mf6_positive_total = float(
    mf6_positive.sum()
)

mf6_negative_total = float(
    mf6_negative.sum()
)


print()
print("=" * 76)
print("GROSS DIRECTIONAL TRANSFER")
print("=" * 76)

print(
    "positive VIC/MF6 =",
    vic_positive,
    mf6_positive_total,
    "m3"
)

print(
    "negative VIC/MF6 =",
    vic_negative,
    mf6_negative_total,
    "m3"
)


if not np.isclose(
    vic_positive,
    mf6_positive_total,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "Positive-volume mapping failed."
    )


if not np.isclose(
    vic_negative,
    mf6_negative_total,
    rtol=1.0e-14,
    atol=1.0e-6,
):

    raise RuntimeError(
        "Negative-volume mapping failed."
    )


print(
    "GROSS SIGNED CONSERVATION: PASS"
)


# ============================================================
# Reverse mapping:
# heterogeneous conceptual MF6 heads -> VIC
#
# Use a spatial head gradient so this is not a trivial
# constant-head test.
# ============================================================

xnorm = np.array([
    (
        x["center_lon"] -
        WEST
    ) /
    (
        EAST -
        WEST
    )
    for x in mf6
])

ynorm = np.array([
    (
        x["center_lat"] -
        SOUTH
    ) /
    (
        NORTH -
        SOUTH
    )
    for x in mf6
])


head_mf6 = (
    -105.328 +
    2.0 *
    (
        xnorm -
        0.5
    ) -
    1.0 *
    (
        ynorm -
        0.5
    )
)


head_vic = (
    A.dot(
        head_mf6
    ) /
    vic_area
)


# ============================================================
# Constant-field preservation
# ============================================================

constant_head = -105.328

constant_mf6 = np.full(
    NM,
    constant_head,
)

constant_vic = (
    A.dot(
        constant_mf6
    ) /
    vic_area
)


constant_error = float(
    np.max(
        np.abs(
            constant_vic -
            constant_head
        )
    )
)


print()
print("=" * 76)
print("MF6 -> VIC HEAD MAPPING")
print("=" * 76)

print(
    "heterogeneous MF6 head range =",
    head_mf6.min(),
    head_mf6.max(),
    "m"
)

print(
    "mapped VIC head range        =",
    head_vic.min(),
    head_vic.max(),
    "m"
)

print(
    "constant-field error         =",
    constant_error,
    "m"
)


if not np.allclose(
    constant_vic,
    constant_head,
    rtol=0.0,
    atol=1.0e-12,
):

    raise RuntimeError(
        "Constant-head preservation failed."
    )


print(
    "CONSTANT HEAD PRESERVATION: PASS"
)


# ============================================================
# Each mapped VIC head must lie within the range of the
# MF6 cells that overlap that VIC cell.
# ============================================================

for i in range(NV):

    js = np.where(
        A[i, :] > 0.0
    )[0]


    hmin = float(
        np.min(
            head_mf6[js]
        )
    )

    hmax = float(
        np.max(
            head_mf6[js]
        )
    )


    if not (
        hmin - 1.0e-12 <=
        head_vic[i] <=
        hmax + 1.0e-12
    ):

        raise RuntimeError(
            f"VIC mapped head {i} "
            "lies outside overlap-head range."
        )


print(
    "LOCAL HEAD RANGE: PASS"
)


# ============================================================
# Area-integral preservation of the head mapping
#
# sum_i H_i A_i
#   =
# sum_j H_j A_j,coupled
# ============================================================

head_integral_vic = float(
    np.sum(
        head_vic *
        vic_area
    )
)

head_integral_mf6 = float(
    np.sum(
        head_mf6 *
        mf6_coupled_area
    )
)


head_integral_error = (
    head_integral_vic -
    head_integral_mf6
)


print(
    "head-area integral error =",
    head_integral_error,
    "m3"
)


if not np.isclose(
    head_integral_vic,
    head_integral_mf6,
    rtol=1.0e-14,
    atol=1.0e-5,
):

    raise RuntimeError(
        "Head area-integral preservation failed."
    )


print(
    "HEAD AREA-INTEGRAL CONSERVATION: PASS"
)


# ============================================================
# Save exchange table
# ============================================================

exchange_rows = []


for i in range(NV):

    for j in range(NM):

        if A[i, j] <= 0.0:
            continue


        exchange_rows.append({
            "vic_id":
                i,

            "vic_row":
                vic[i]["row"],

            "vic_col":
                vic[i]["col"],

            "mf6_id":
                j,

            "mf6_row":
                mf6[j]["row"],

            "mf6_col":
                mf6[j]["col"],

            "overlap_area_m2":
                A[i, j],

            "fraction_of_vic_cell":
                A[i, j] /
                vic_area[i],

            "fraction_of_mf6_coupled_area":
                (
                    A[i, j] /
                    mf6_coupled_area[j]
                    if mf6_coupled_area[j] > 0.0
                    else np.nan
                ),
        })


with (
    OUT /
    "exchange_table.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            exchange_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        exchange_rows
    )


# ============================================================
# Save actual VIC cells
# ============================================================

vic_rows = []


for i, x in enumerate(vic):

    vic_rows.append({
        "vic_id":
            i,

        "row":
            x["row"],

        "col":
            x["col"],

        "lon":
            x["lon"],

        "lat":
            x["lat"],

        "area_m2":
            x["vic_area_m2"],

        "q_mm":
            x["q_mm"],

        "volume_m3":
            vic_volume_m3[i],

        "mapped_mf6_head_m":
            head_vic[i],
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
            vic_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        vic_rows
    )


# ============================================================
# Save conceptual MF6 cells
# ============================================================

mf6_rows = []


for j, x in enumerate(mf6):

    mf6_rows.append({
        "mf6_id":
            j,

        "row":
            x["row"],

        "col":
            x["col"],

        "center_lon":
            x["center_lon"],

        "center_lat":
            x["center_lat"],

        "rectangle_area_m2":
            x["rectangle_area_m2"],

        "coupled_area_m2":
            mf6_coupled_area[j],

        "head_m":
            head_mf6[j],

        "mapped_net_volume_m3":
            mf6_net_volume_m3[j],

        "mapped_positive_volume_m3":
            mf6_positive[j],

        "mapped_negative_volume_m3":
            mf6_negative[j],
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
            mf6_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        mf6_rows
    )


# ============================================================
# Summary
# ============================================================

summary = {
    "active_vic_cells":
        NV,

    "conceptual_mf6_cells":
        NM,

    "overlap_records":
        len(exchange_rows),

    "total_vic_area_m2":
        float(
            vic_area.sum()
        ),

    "total_overlap_area_m2":
        float(
            A.sum()
        ),

    "max_vic_coverage_error_m2":
        max_coverage_error,

    "vic_total_exchange_m3":
        vic_total,

    "mf6_total_exchange_m3":
        mf6_total,

    "net_conservation_error_m3":
        net_error,

    "vic_positive_exchange_m3":
        vic_positive,

    "mf6_positive_exchange_m3":
        mf6_positive_total,

    "vic_negative_exchange_m3":
        vic_negative,

    "mf6_negative_exchange_m3":
        mf6_negative_total,

    "constant_head_error_m":
        constant_error,

    "head_area_integral_error":
        head_integral_error,
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


print()
print("=" * 76)
print("H7b REAL STEHEKIN MISMATCHED-GRID MAPPER: PASS")
print("=" * 76)

print()
print("overlap records =",
      len(exchange_rows))

print()
print("Saved:")
print(OUT / "exchange_table.csv")
print(OUT / "vic_cells.csv")
print(OUT / "mf6_cells.csv")
print(OUT / "summary.csv")
