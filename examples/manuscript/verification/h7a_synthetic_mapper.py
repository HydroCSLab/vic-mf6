#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv

import numpy as np


OUT = Path(
    str(RESULT_ROOT / 'H7a_synthetic_mapper')
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Rectangle grids
#
# Same total domain:
#
#     x = 0 .. 4
#     y = 0 .. 4
#
# VIC:
#     2 x 2 regular grid
#
# MF6:
#     3 x 2 irregular grid
#
# Neither internal x nor y boundaries match.
# ============================================================

vic_x = np.array([
    0.0,
    2.0,
    4.0,
])

vic_y = np.array([
    0.0,
    2.0,
    4.0,
])


mf6_x = np.array([
    0.0,
    1.0,
    3.0,
    4.0,
])

mf6_y = np.array([
    0.0,
    1.5,
    4.0,
])


def make_cells(
    xedges,
    yedges,
    prefix,
):

    cells = []

    cid = 0

    for iy in range(
        len(yedges) - 1
    ):

        for ix in range(
            len(xedges) - 1
        ):

            xmin = xedges[ix]
            xmax = xedges[ix + 1]

            ymin = yedges[iy]
            ymax = yedges[iy + 1]

            area = (
                (xmax - xmin) *
                (ymax - ymin)
            )

            cells.append({
                "id":
                    f"{prefix}{cid}",

                "index":
                    cid,

                "xmin":
                    xmin,

                "xmax":
                    xmax,

                "ymin":
                    ymin,

                "ymax":
                    ymax,

                "area":
                    area,
            })

            cid += 1

    return cells


vic = make_cells(
    vic_x,
    vic_y,
    "V",
)

mf6 = make_cells(
    mf6_x,
    mf6_y,
    "M",
)


def intersection_area(
    a,
    b,
):

    dx = max(
        0.0,
        min(
            a["xmax"],
            b["xmax"]
        ) -
        max(
            a["xmin"],
            b["xmin"]
        )
    )

    dy = max(
        0.0,
        min(
            a["ymax"],
            b["ymax"]
        ) -
        max(
            a["ymin"],
            b["ymin"]
        )
    )

    return (
        dx *
        dy
    )


# ============================================================
# Build overlap matrix
#
# rows = VIC
# cols = MF6
# ============================================================

A = np.zeros(
    (
        len(vic),
        len(mf6)
    ),
    dtype=float,
)


table = []


for i, vc in enumerate(vic):

    for j, mc in enumerate(mf6):

        overlap = intersection_area(
            vc,
            mc
        )

        if overlap <= 0.0:
            continue

        A[i, j] = overlap

        table.append({
            "vic_id":
                vc["id"],

            "mf6_id":
                mc["id"],

            "overlap_area_m2":
                overlap,

            "vic_area_m2":
                vc["area"],

            "mf6_area_m2":
                mc["area"],

            "vic_fraction":
                overlap /
                vc["area"],

            "mf6_fraction":
                overlap /
                mc["area"],
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
            table[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        table
    )


print()
print("===== H7a OVERLAP MATRIX =====")
print(A)


# ============================================================
# Coverage validation
# ============================================================

vic_area = np.array([
    x["area"]
    for x in vic
])

mf6_area = np.array([
    x["area"]
    for x in mf6
])


vic_covered = A.sum(
    axis=1
)

mf6_covered = A.sum(
    axis=0
)


print()
print("VIC areas:")
print(vic_area)

print("VIC covered:")
print(vic_covered)

print()
print("MF6 areas:")
print(mf6_area)

print("MF6 covered:")
print(mf6_covered)


if not np.allclose(
    vic_covered,
    vic_area,
    rtol=0.0,
    atol=1.0e-14,
):

    raise RuntimeError(
        "VIC domain is not completely "
        "covered by MF6."
    )


if not np.allclose(
    mf6_covered,
    mf6_area,
    rtol=0.0,
    atol=1.0e-14,
):

    raise RuntimeError(
        "MF6 domain is not completely "
        "covered by VIC."
    )


print()
print("DOMAIN COVERAGE: PASS")


# ============================================================
# TEST 1:
# heterogeneous signed VIC -> MF6 exchange
# ============================================================

# [mm over coupling interval]
#
# Deliberately includes both directions.
#
# positive = VIC -> MF6
# negative = MF6 -> VIC

q_vic_mm = np.array([
    7.0,
    -4.0,
    2.5,
    -1.5,
])


# Direct volume from VIC cells.
vic_volume_m3 = (
    q_vic_mm *
    1.0e-3 *
    vic_area
)


vic_total_m3 = float(
    vic_volume_m3.sum()
)


# Map to MF6 through overlap areas.
mf6_volume_m3 = (
    1.0e-3 *
    A.T.dot(
        q_vic_mm
    )
)


mf6_total_m3 = float(
    mf6_volume_m3.sum()
)


net_error = (
    mf6_total_m3 -
    vic_total_m3
)


print()
print("=" * 72)
print("TEST 1: SIGNED VIC -> MF6 VOLUME")
print("=" * 72)

print(
    "VIC cell volumes:",
    vic_volume_m3
)

print(
    "MF6 cell volumes:",
    mf6_volume_m3
)

print(
    "VIC total =",
    vic_total_m3,
    "m3"
)

print(
    "MF6 total =",
    mf6_total_m3,
    "m3"
)

print(
    "error     =",
    net_error,
    "m3"
)


if not np.isclose(
    mf6_total_m3,
    vic_total_m3,
    rtol=0.0,
    atol=1.0e-14,
):

    raise RuntimeError(
        "Net-volume conservation failed."
    )


# ============================================================
# TEST 2:
# positive and negative volumes separately
# ============================================================

positive_direct = float(
    np.sum(
        np.maximum(
            q_vic_mm,
            0.0
        ) *
        1.0e-3 *
        vic_area
    )
)


negative_direct = float(
    np.sum(
        np.minimum(
            q_vic_mm,
            0.0
        ) *
        1.0e-3 *
        vic_area
    )
)


positive_overlap = float(
    np.sum(
        1.0e-3 *
        A.T.dot(
            np.maximum(
                q_vic_mm,
                0.0
            )
        )
    )
)


negative_overlap = float(
    np.sum(
        1.0e-3 *
        A.T.dot(
            np.minimum(
                q_vic_mm,
                0.0
            )
        )
    )
)


print()
print("=" * 72)
print("TEST 2: GROSS SIGNED TRANSFER")
print("=" * 72)

print(
    "positive direct/overlap =",
    positive_direct,
    positive_overlap
)

print(
    "negative direct/overlap =",
    negative_direct,
    negative_overlap
)


if not np.isclose(
    positive_direct,
    positive_overlap,
    rtol=0.0,
    atol=1.0e-14,
):

    raise RuntimeError(
        "Positive transfer was not conserved."
    )


if not np.isclose(
    negative_direct,
    negative_overlap,
    rtol=0.0,
    atol=1.0e-14,
):

    raise RuntimeError(
        "Negative transfer was not conserved."
    )


print("GROSS SIGNED CONSERVATION: PASS")


# ============================================================
# TEST 3:
# MF6 head -> VIC area-weighted head
# ============================================================

head_mf6 = np.array([
    -100.0,
    -110.0,
    -120.0,
    -130.0,
    -115.0,
    -95.0,
])


head_vic = (
    A.dot(
        head_mf6
    ) /
    vic_covered
)


print()
print("=" * 72)
print("TEST 3: MF6 -> VIC HEAD")
print("=" * 72)

print(
    "MF6 heads:",
    head_mf6
)

print(
    "mapped VIC heads:",
    head_vic
)


# Each averaged head must lie inside the range
# of the overlapping MF6 heads.

for i in range(
    len(vic)
):

    js = np.where(
        A[i] > 0.0
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
        hmin <=
        head_vic[i] <=
        hmax
    ):

        raise RuntimeError(
            f"Mapped VIC head {i} "
            "lies outside overlap range."
        )


print("HEAD WEIGHTING: PASS")


# ============================================================
# TEST 4:
#
# Prove that mapping MF6 head to VIC first is equivalent
# to evaluating current linear groundwater exchange over
# every VIC-MF6 intersection.
#
# q = C_i * (Hsoil_i - Hgw)
#
# C_i is a VIC-cell-specific exchange coefficient.
# ============================================================

soil_head_vic = np.array([
    -105.0,
    -108.0,
    -112.0,
    -101.0,
])


# arbitrary positive conductance per area:
# [m/day per m head difference]
conductance = np.array([
    1.0e-4,
    2.0e-4,
    1.5e-4,
    0.8e-4,
])


# Direct intersection-level integration.
direct_flux_volume = 0.0


for i in range(
    len(vic)
):

    for j in range(
        len(mf6)
    ):

        if A[i, j] <= 0.0:
            continue

        q_m_day = (
            conductance[i] *
            (
                soil_head_vic[i] -
                head_mf6[j]
            )
        )

        direct_flux_volume += (
            q_m_day *
            A[i, j]
        )


# First area-average MF6 head onto VIC,
# then use the VIC exchange formulation.

mapped_flux_volume = float(
    np.sum(
        conductance *
        (
            soil_head_vic -
            head_vic
        ) *
        vic_area
    )
)


flux_equivalence_error = (
    mapped_flux_volume -
    direct_flux_volume
)


print()
print("=" * 72)
print("TEST 4: HEAD-MAPPING FLUX EQUIVALENCE")
print("=" * 72)

print(
    "intersection integration =",
    direct_flux_volume,
    "m3/day"
)

print(
    "mapped-head integration  =",
    mapped_flux_volume,
    "m3/day"
)

print(
    "difference               =",
    flux_equivalence_error,
    "m3/day"
)


if not np.isclose(
    mapped_flux_volume,
    direct_flux_volume,
    rtol=1.0e-14,
    atol=1.0e-14,
):

    raise RuntimeError(
        "Area-weighted head mapping is not "
        "flux-equivalent."
    )


print(
    "LINEAR HEAD-FLUX EQUIVALENCE: PASS"
)


# ============================================================
# Save mapped values
# ============================================================

rows = []

for j, mc in enumerate(mf6):

    rows.append({
        "mf6_id":
            mc["id"],

        "area_m2":
            mc["area"],

        "mapped_volume_m3":
            mf6_volume_m3[j],

        "head_m":
            head_mf6[j],
    })


with (
    OUT /
    "mf6_mapping.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


summary = {
    "vic_cells":
        len(vic),

    "mf6_cells":
        len(mf6),

    "overlap_records":
        int(
            np.count_nonzero(A)
        ),

    "total_domain_area_m2":
        float(
            vic_area.sum()
        ),

    "vic_total_exchange_m3":
        vic_total_m3,

    "mf6_total_exchange_m3":
        mf6_total_m3,

    "net_conservation_error_m3":
        net_error,

    "positive_transfer_error_m3":
        positive_overlap -
        positive_direct,

    "negative_transfer_error_m3":
        negative_overlap -
        negative_direct,

    "head_flux_equivalence_error_m3_day":
        flux_equivalence_error,
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
print("=" * 72)
print("H7a SYNTHETIC CONSERVATIVE MAPPER: PASS")
print("=" * 72)

print()
print("Saved:")
print(OUT / "exchange_table.csv")
print(OUT / "mf6_mapping.csv")
print(OUT / "summary.csv")
