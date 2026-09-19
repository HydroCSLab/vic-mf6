#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv

import netCDF4 as nc
import numpy as np


ROOT = Path(
    str(RESULT_ROOT)
)

DOMAIN = Path(
    str(SAMPLE_ROOT / 'parameters/domain.stehekin.20151028.nc')
)


# ============================================================
# Locate latest H3a and H3b
# ============================================================

h3a_dirs = sorted(
    ROOT.glob("H3a_restart_equivalence_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

h3b_dirs = sorted(
    ROOT.glob("H3b_closed_loop_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)


if not h3a_dirs:
    raise RuntimeError("No H3a result found.")

if not h3b_dirs:
    raise RuntimeError("No H3b result found.")


H3A = h3a_dirs[0]
H3B = h3b_dirs[0]

OUT = ROOT / "H3c_feedback_analysis"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)


print("H3a:", H3A)
print("H3b:", H3B)


# ============================================================
# Read VIC domain areas
# ============================================================

with nc.Dataset(DOMAIN) as ds:

    area_var = ds["area"]

    area = np.ma.asarray(
        area_var[:],
        dtype=float,
    )

    mask = (
        np.asarray(ds["mask"][:]) > 0
    )

    units = (
        getattr(area_var, "units", "")
        .strip()
        .lower()
        .replace(" ", "")
    )


if units in ("m2", "m^2", "m**2"):

    area_m2 = area

elif units in ("km2", "km^2", "km**2"):

    area_m2 = area * 1.0e6

else:

    raise RuntimeError(
        f"Unsupported area units: {units!r}"
    )


active = (
    mask &
    ~np.ma.getmaskarray(area_m2)
)

A = np.asarray(area_m2)[active]

TOTAL_AREA = float(
    np.sum(A)
)


# ============================================================
# Locate the ten daily VIC files
# ============================================================

def daily_files(root):

    files = sorted(
        root.glob(
            "vic/day_*/results/fluxes*.nc"
        )
    )

    if not files:

        files = sorted(
            root.glob(
                "segmented/day_*/results/fluxes*.nc"
            )
        )

    if len(files) != 10:

        raise RuntimeError(
            f"Expected 10 daily files in {root}; "
            f"found {len(files)}"
        )

    return files


a_files = daily_files(H3A)
b_files = daily_files(H3B)


# ============================================================
# Read H3b MF6 trajectory
# ============================================================

with (
    H3B /
    "coupling_daily.csv"
).open() as f:

    b_coupling = list(
        csv.DictReader(f)
    )


if len(b_coupling) != 10:

    raise RuntimeError(
        "Expected 10 rows in H3b coupling_daily.csv"
    )


# ============================================================
# Area-weighted field helpers
# ============================================================

def weighted_mean(field):

    f = np.ma.asarray(field)

    valid = (
        active &
        ~np.ma.getmaskarray(f)
    )

    av = np.asarray(area_m2)[valid]
    fv = np.asarray(f)[valid]

    return float(
        np.sum(fv * av) /
        np.sum(av)
    )


def integrated_volume_mm(field):

    f = np.ma.asarray(field)

    valid = (
        active &
        ~np.ma.getmaskarray(f)
    )

    av = np.asarray(area_m2)[valid]
    fv = np.asarray(f)[valid]

    return float(
        np.sum(
            fv *
            1.0e-3 *
            av
        )
    )


# ============================================================
# Daily comparison
# ============================================================

rows = []

cum_a = 0.0
cum_b = 0.0


for n, (fa, fb) in enumerate(
    zip(a_files, b_files),
    start=1,
):

    with nc.Dataset(fa) as a, \
         nc.Dataset(fb) as b:

        qa = np.ma.asarray(
            a["OUT_GW_EXCHANGE"][0]
        )

        qb = np.ma.asarray(
            b["OUT_GW_EXCHANGE"][0]
        )

        soil_a = np.ma.asarray(
            a["OUT_SOIL_MOIST"][0]
        )

        soil_b = np.ma.asarray(
            b["OUT_SOIL_MOIST"][0]
        )


        qa_aw = weighted_mean(qa)
        qb_aw = weighted_mean(qb)

        Va = integrated_volume_mm(qa)
        Vb = integrated_volume_mm(qb)

        cum_a += Va
        cum_b += Vb


        bottom_a = weighted_mean(
            soil_a[-1]
        )

        bottom_b = weighted_mean(
            soil_b[-1]
        )


        h_before = float(
            b_coupling[n - 1][
                "mf6_head_before_m"
            ]
        )

        h_after = float(
            b_coupling[n - 1][
                "mf6_head_after_m"
            ]
        )


        rows.append({
            "day":
                n,

            "mf6_head_before_m":
                h_before,

            "mf6_head_after_m":
                h_after,

            "fixed_q_mm_day":
                qa_aw,

            "coupled_q_mm_day":
                qb_aw,

            "q_feedback_difference_mm_day":
                qb_aw - qa_aw,

            "fixed_volume_m3":
                Va,

            "coupled_volume_m3":
                Vb,

            "daily_volume_feedback_m3":
                Vb - Va,

            "cumulative_fixed_volume_m3":
                cum_a,

            "cumulative_coupled_volume_m3":
                cum_b,

            "cumulative_feedback_volume_m3":
                cum_b - cum_a,

            "fixed_bottom_moist_mm":
                bottom_a,

            "coupled_bottom_moist_mm":
                bottom_b,

            "bottom_moist_difference_mm":
                bottom_b - bottom_a,
        })


# ============================================================
# Save daily table
# ============================================================

csv_file = (
    OUT /
    "feedback_daily.csv"
)

with csv_file.open(
    "w",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    w.writeheader()
    w.writerows(rows)


# ============================================================
# Summary
# ============================================================

total_fixed = rows[-1][
    "cumulative_fixed_volume_m3"
]

total_coupled = rows[-1][
    "cumulative_coupled_volume_m3"
]

feedback_volume = (
    total_coupled -
    total_fixed
)


feedback_percent = (
    100.0 *
    feedback_volume /
    abs(total_fixed)
    if total_fixed != 0.0
    else np.nan
)


q_day1_fixed = rows[0][
    "fixed_q_mm_day"
]

q_day1_coupled = rows[0][
    "coupled_q_mm_day"
]

q_day10_fixed = rows[-1][
    "fixed_q_mm_day"
]

q_day10_coupled = rows[-1][
    "coupled_q_mm_day"
]


print()
print("=" * 76)
print("H3c FEEDBACK ISOLATION")
print("=" * 76)

print()
print("DAY 1")
print(
    "fixed-head q   =",
    q_day1_fixed,
    "mm/day"
)
print(
    "coupled q      =",
    q_day1_coupled,
    "mm/day"
)
print(
    "difference     =",
    q_day1_coupled -
    q_day1_fixed,
    "mm/day"
)

print()
print("DAY 10")
print(
    "fixed-head q   =",
    q_day10_fixed,
    "mm/day"
)
print(
    "coupled q      =",
    q_day10_coupled,
    "mm/day"
)
print(
    "difference     =",
    q_day10_coupled -
    q_day10_fixed,
    "mm/day"
)

print()
print("10-DAY CUMULATIVE")

print(
    "fixed-head VIC volume =",
    total_fixed,
    "m3"
)

print(
    "coupled VIC volume    =",
    total_coupled,
    "m3"
)

print(
    "feedback difference   =",
    feedback_volume,
    "m3"
)

print(
    "feedback difference   =",
    feedback_percent,
    "% of |fixed|"
)

print()
print(
    "MF6 final head        =",
    rows[-1]["mf6_head_after_m"],
    "m"
)

print(
    "MF6 total dh          =",
    rows[-1]["mf6_head_after_m"]
    - rows[0]["mf6_head_before_m"],
    "m"
)


# ============================================================
# Internal consistency:
# Day 1 MUST be identical because H3a and H3b begin with the
# same head and same VIC state.
# ============================================================

day1_pass = np.isclose(
    q_day1_fixed,
    q_day1_coupled,
    rtol=0.0,
    atol=1.0e-12,
)


print()
print(
    "DAY-1 CONTROL:",
    "PASS"
    if day1_pass
    else "FAIL"
)


# ============================================================
# Save summary
# ============================================================

summary = {
    "fixed_total_volume_m3":
        total_fixed,

    "coupled_total_volume_m3":
        total_coupled,

    "feedback_volume_difference_m3":
        feedback_volume,

    "feedback_percent_of_fixed":
        feedback_percent,

    "day1_fixed_q_mm_day":
        q_day1_fixed,

    "day1_coupled_q_mm_day":
        q_day1_coupled,

    "day10_fixed_q_mm_day":
        q_day10_fixed,

    "day10_coupled_q_mm_day":
        q_day10_coupled,

    "final_mf6_head_m":
        rows[-1][
            "mf6_head_after_m"
        ],

    "mf6_head_change_m":
        rows[-1][
            "mf6_head_after_m"
        ] -
        rows[0][
            "mf6_head_before_m"
        ],

    "day1_control_pass":
        bool(day1_pass),
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
        fieldnames=summary.keys()
    )

    w.writeheader()
    w.writerow(summary)


print()
print("Saved:")
print(csv_file)
print(OUT / "summary.csv")
