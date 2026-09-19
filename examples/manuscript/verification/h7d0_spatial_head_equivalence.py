#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv
import os
import subprocess
import time

import netCDF4 as nc
import numpy as np


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

VIC_CSV = (
    H7B /
    "vic_cells.csv"
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    ROOT /
    f"H7d0_spatial_head_equivalence_{STAMP}"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Read H7b VIC cells
# ============================================================

with VIC_CSV.open() as f:

    cells = list(
        csv.DictReader(f)
    )


if len(cells) != 16:

    raise RuntimeError(
        f"Expected 16 VIC cells; found {len(cells)}"
    )


for row in cells:

    row["row"] = int(
        row["row"]
    )

    row["col"] = int(
        row["col"]
    )

    row["lat"] = float(
        row["lat"]
    )

    row["lon"] = float(
        row["lon"]
    )

    row["head"] = float(
        row["mapped_mf6_head_m"]
    )


# ============================================================
# Spatial-head file
# ============================================================

HEAD_FILE = (
    OUT /
    "vic_gw_head_offsets.txt"
)


with HEAD_FILE.open("w") as f:

    f.write(
        "# latitude longitude head_offset_m\n"
    )

    for row in cells:

        f.write(
            f"{row['lat']:.17g} "
            f"{row['lon']:.17g} "
            f"{row['head']:.17g}\n"
        )


print()
print("===== H7d0 SPATIAL HEAD FILE =====")
print(HEAD_FILE)

for row in cells:

    print(
        f"cell ({row['row']},{row['col']}) "
        f"lat={row['lat']:.6f} "
        f"lon={row['lon']:.6f} "
        f"head={row['head']:.9f}"
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


base = []

for line in BASE_GLOBAL.read_text().splitlines():

    s = line.strip()

    if (
        s and
        not s.startswith("#")
    ):

        if s.split()[0] in controlled:
            continue

    base.append(line)


BASE_TEXT = (
    "\n".join(base) +
    "\n"
)


def make_global(
    path,
    result_dir,
):

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        BASE_TEXT +
        f"""
STARTYEAR    1949
STARTMONTH   1
STARTDAY     1
STARTSEC     0

NRECS        24

RESULT_DIR   {result_dir}

AGGFREQ      NHOURS 24
"""
    )


# ============================================================
# Runner
# ============================================================

def run_vic(
    label,
    spatial_file=None,
    scalar_head=None,
):

    run_dir = (
        OUT /
        label
    )

    result_dir = (
        run_dir /
        "results"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    global_file = (
        run_dir /
        "run.global.txt"
    )


    make_global(
        global_file,
        result_dir,
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
            "0.001",

        "VIC_GW_MAX_DRAIN_FRACTION":
            "1.0",

        "VIC_GW_EXCHANGE_LENGTH_M":
            "100.0",
    })


    env.pop(
        "VIC_GW_HEAD_FILE",
        None
    )

    env.pop(
        "VIC_GW_HEAD_OFFSET_M",
        None
    )


    if spatial_file is not None:

        env[
            "VIC_GW_HEAD_FILE"
        ] = str(
            spatial_file
        )


    elif scalar_head is not None:

        env[
            "VIC_GW_HEAD_OFFSET_M"
        ] = (
            f"{scalar_head:.17g}"
        )


    else:

        raise RuntimeError(
            "Need spatial_file or scalar_head."
        )


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
            f"VIC failed for {label}; "
            f"see {run_dir / 'stderr.log'}"
        )


    files = sorted(
        result_dir.glob(
            "fluxes*.nc"
        )
    )


    if len(files) != 1:

        raise RuntimeError(
            f"Expected one flux file for {label}"
        )


    return files[0]


# ============================================================
# Spatial run
# ============================================================

print()
print("===== RUN SPATIAL HEAD FIELD =====")

SPATIAL_FLUX = run_vic(
    "spatial",
    spatial_file=
        HEAD_FILE,
)

print(
    "[OK]",
    SPATIAL_FLUX
)


# ============================================================
# Scalar controls
# ============================================================

scalar_flux = {}


unique_heads = sorted(
    set(
        row["head"]
        for row in cells
    )
)


print()
print(
    "scalar control runs =",
    len(unique_heads)
)


for k, head in enumerate(
    unique_heads,
    1
):

    print(
        f"[{k:02d}/{len(unique_heads):02d}] "
        f"head={head:.9f}"
    )


    scalar_flux[head] = run_vic(
        f"scalar_{k:02d}",
        scalar_head=
            head,
    )


# ============================================================
# Exact comparison helper
# ============================================================

def exact_equal(
    a,
    b,
):

    a = np.ma.asarray(a)
    b = np.ma.asarray(b)


    if a.shape != b.shape:
        return False


    if not np.array_equal(
        np.ma.getmaskarray(a),
        np.ma.getmaskarray(b),
    ):

        return False


    return np.array_equal(
        a.compressed(),
        b.compressed(),
        equal_nan=True,
    )


def absmax(
    a,
    b,
):

    a = np.ma.asarray(a)
    b = np.ma.asarray(b)

    aa = a.compressed()
    bb = b.compressed()

    if aa.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(
                aa -
                bb
            )
        )
    )


TEST_VARS = [
    "OUT_GW_EXCHANGE",
    "OUT_SOIL_MOIST",
    "OUT_RUNOFF",
    "OUT_WATER_ERROR",
]


# ============================================================
# Compare each spatial cell against scalar control
# ============================================================

comparison = []


with nc.Dataset(
    SPATIAL_FLUX
) as spatial:

    for cell in cells:

        scalar_path = (
            scalar_flux[
                cell["head"]
            ]
        )


        with nc.Dataset(
            scalar_path
        ) as scalar:

            for var in TEST_VARS:

                a = np.ma.asarray(
                    spatial[var][
                        ...,
                        cell["row"],
                        cell["col"]
                    ]
                )

                b = np.ma.asarray(
                    scalar[var][
                        ...,
                        cell["row"],
                        cell["col"]
                    ]
                )


                same = exact_equal(
                    a,
                    b,
                )

                diff = absmax(
                    a,
                    b,
                )


                comparison.append({
                    "row":
                        cell["row"],

                    "col":
                        cell["col"],

                    "lat":
                        cell["lat"],

                    "lon":
                        cell["lon"],

                    "head_offset_m":
                        cell["head"],

                    "variable":
                        var,

                    "bitwise_identical":
                        same,

                    "absmax_difference":
                        diff,
                })


# ============================================================
# Save
# ============================================================

with (
    OUT /
    "comparison.csv"
).open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            comparison[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        comparison
    )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 88)
print("H7d0 SPATIAL-HEAD EQUIVALENCE")
print("=" * 88)


overall = True


for cell in cells:

    subset = [
        r
        for r in comparison
        if (
            r["row"] ==
            cell["row"] and
            r["col"] ==
            cell["col"]
        )
    ]


    passed = all(
        r[
            "bitwise_identical"
        ]
        for r in subset
    )


    overall &= passed


    maxdiff = max(
        r[
            "absmax_difference"
        ]
        for r in subset
    )


    print(
        f"cell ({cell['row']},{cell['col']}) "
        f"H={cell['head']:.9f} "
        f"{'PASS' if passed else 'FAIL'} "
        f"absmax={maxdiff:.17g}"
    )


print()
print(
    "cells tested     =",
    len(cells)
)

print(
    "variables/cell   =",
    len(TEST_VARS)
)

print(
    "comparisons      =",
    len(comparison)
)


print()
print("=" * 88)

print(
    "H7d0 SPATIAL GROUNDWATER HEAD INTERFACE:",
    "PASS"
    if overall
    else "REQUIRES REVIEW"
)

print("=" * 88)


print()
print("Saved:")
print(HEAD_FILE)
print(OUT / "comparison.csv")

print()
print("Experiment directory:")
print(OUT)
