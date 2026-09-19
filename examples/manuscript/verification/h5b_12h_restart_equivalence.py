#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
from datetime import date
import csv
import os
import subprocess
import time

import netCDF4 as nc
import numpy as np


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

EXPROOT = Path(
    str(RESULT_ROOT)
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    EXPROOT /
    f"H5b_12h_restart_equivalence_{STAMP}"
)

GLOBALS = OUT / "globals"
LOGS = OUT / "logs"

CONT = OUT / "continuous"
SEG1 = OUT / "half_1"
SEG2 = OUT / "half_2"

for p in (
    GLOBALS,
    LOGS,
    CONT,
    SEG1,
    SEG2,
):
    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Common experimental environment
# ============================================================

ENV = os.environ.copy()

ENV.update({
    "OMP_NUM_THREADS":
        "1",

    # Required only for the two 12-hour segments.
    "VIC_ALLOW_PARTIAL_DAY":
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

    "VIC_GW_HEAD_OFFSET_M":
        "-105.328",

    "VIC_GW_EXCHANGE_LENGTH_M":
        "100.0",
})


# ============================================================
# Preserve provenance
# ============================================================

def capture(cmd):
    from manuscript_paths import source_provenance
    return source_provenance(cmd)


(OUT / "vic_commit.txt").write_text(
    capture([
        "git",
        "rev-parse",
        "HEAD",
    ])
)

(OUT / "vic_status.txt").write_text(
    capture([
        "git",
        "status",
        "--short",
    ])
)

(OUT / "vic_diff.txt").write_text(
    capture([
        "git",
        "diff",
    ])
)


# ============================================================
# Remove timing/state/output controls from canonical global
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


lines = []

for line in BASE_GLOBAL.read_text().splitlines():

    stripped = line.strip()

    if (
        stripped and
        not stripped.startswith("#")
    ):

        key = stripped.split()[0]

        if key in controlled:
            continue

    lines.append(line)


BASE_TEXT = (
    "\n".join(lines) +
    "\n"
)


# ============================================================
# Global-file builder
# ============================================================

def make_global(
    path,
    start_sec,
    nrecs,
    result_dir,
    state_prefix,
    state_year,
    state_month,
    state_day,
    state_sec,
    init_state=None,
):

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    state_prefix.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    extra = [
        "",
        "# ================================================",
        "# H5b subdaily restart-equivalence experiment",
        "# ================================================",

        "STARTYEAR    1949",
        "STARTMONTH   1",
        "STARTDAY     1",
        f"STARTSEC     {start_sec}",

        f"NRECS        {nrecs}",

        f"RESULT_DIR   {result_dir}",

        # Align output boundaries with the restart boundary.
        "AGGFREQ      NHOURS 12",
    ]


    if init_state is not None:

        extra.append(
            f"INIT_STATE   {init_state}"
        )


    extra.extend([
        f"STATENAME    {state_prefix}",

        f"STATEYEAR    {state_year}",
        f"STATEMONTH   {state_month}",
        f"STATEDAY     {state_day}",
        f"STATESEC     {state_sec}",

        "STATE_FORMAT NETCDF4_CLASSIC",
        "",
    ])


    path.write_text(
        BASE_TEXT +
        "\n".join(extra)
    )


# ============================================================
# VIC runner
# ============================================================

def run_vic(
    label,
    global_file,
):

    stdout = (
        LOGS /
        f"{label}.stdout.log"
    )

    stderr = (
        LOGS /
        f"{label}.stderr.log"
    )


    print()
    print("=" * 72)
    print(label)
    print("=" * 72)


    with stdout.open("w") as out, \
         stderr.open("w") as err:

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
            env=ENV,
            stdout=out,
            stderr=err,
        )


    if proc.returncode != 0:

        print("[FAILED]", label)
        print("stderr:", stderr)

        raise RuntimeError(
            f"VIC failed: {label}"
        )


    print("[OK]", label)


def find_one(
    directory,
    pattern,
):

    files = sorted(
        directory.glob(pattern)
    )

    if len(files) != 1:

        raise RuntimeError(
            f"Expected exactly one {pattern} in "
            f"{directory}; found {files}"
        )

    return files[0]


# ============================================================
# 1. Continuous 24-hour run
#
# 1949-01-01 00:00
#     ->
# 1949-01-02 00:00
# ============================================================

cont_results = (
    CONT /
    "results"
)

cont_state_dir = (
    CONT /
    "state"
)

cont_global = (
    GLOBALS /
    "continuous_24h.global.txt"
)


make_global(
    path=
        cont_global,

    start_sec=
        0,

    nrecs=
        24,

    result_dir=
        cont_results,

    state_prefix=
        cont_state_dir / "state",

    state_year=
        1949,

    state_month=
        1,

    state_day=
        2,

    state_sec=
        0,
)


run_vic(
    "continuous_24h",
    cont_global
)


cont_flux = find_one(
    cont_results,
    "fluxes*.nc"
)

cont_state = find_one(
    cont_state_dir,
    "state*.nc"
)


# ============================================================
# 2. First 12-hour segment
#
# 00:00 -> 12:00
# ============================================================

seg1_results = (
    SEG1 /
    "results"
)

seg1_state_dir = (
    SEG1 /
    "state"
)

seg1_global = (
    GLOBALS /
    "half_1.global.txt"
)


make_global(
    path=
        seg1_global,

    start_sec=
        0,

    nrecs=
        12,

    result_dir=
        seg1_results,

    state_prefix=
        seg1_state_dir / "state",

    state_year=
        1949,

    state_month=
        1,

    state_day=
        1,

    state_sec=
        43200,
)


run_vic(
    "half_1_00000_to_43200",
    seg1_global
)


seg1_flux = find_one(
    seg1_results,
    "fluxes*.nc"
)

seg1_state = find_one(
    seg1_state_dir,
    "state*.nc"
)


# ============================================================
# 3. Second 12-hour segment
#
# 12:00 -> next midnight
# ============================================================

seg2_results = (
    SEG2 /
    "results"
)

seg2_state_dir = (
    SEG2 /
    "state"
)

seg2_global = (
    GLOBALS /
    "half_2.global.txt"
)


make_global(
    path=
        seg2_global,

    start_sec=
        43200,

    nrecs=
        12,

    result_dir=
        seg2_results,

    state_prefix=
        seg2_state_dir / "state",

    state_year=
        1949,

    state_month=
        1,

    state_day=
        2,

    state_sec=
        0,

    init_state=
        seg1_state,
)


run_vic(
    "half_2_43200_to_86400",
    seg2_global
)


seg2_flux = find_one(
    seg2_results,
    "fluxes*.nc"
)

seg2_state = find_one(
    seg2_state_dir,
    "state*.nc"
)


# ============================================================
# Comparison helpers
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


    av = a.compressed()
    bv = b.compressed()


    if (
        av.dtype.kind in "fc" and
        bv.dtype.kind in "fc"
    ):

        return np.array_equal(
            av,
            bv,
            equal_nan=True,
        )


    return np.array_equal(
        av,
        bv,
    )


def absmax_difference(
    a,
    b,
):

    a = np.ma.asarray(a)
    b = np.ma.asarray(b)


    if a.shape != b.shape:
        return np.nan


    aa = a.compressed()
    bb = b.compressed()


    if aa.size == 0:
        return 0.0


    if (
        aa.dtype.kind not in "fciu" or
        bb.dtype.kind not in "fciu"
    ):
        return np.nan


    return float(
        np.max(
            np.abs(
                aa -
                bb
            )
        )
    )


# ============================================================
# 4. Compare output records
#
# Continuous should contain two 12-hour records.
#
# Segmented:
#
#     half 1 -> one record
#     half 2 -> one record
#
# Concatenate them along time.
# ============================================================

print()
print("=" * 72)
print("H5b OUTPUT EQUIVALENCE")
print("=" * 72)


comparison = []


with nc.Dataset(
    cont_flux
) as c, \
     nc.Dataset(
         seg1_flux
     ) as a, \
     nc.Dataset(
         seg2_flux
     ) as b:


    # --------------------------------------------------------
    # Time coordinate check
    # --------------------------------------------------------

    if (
        "time" in c.variables and
        "time" in a.variables and
        "time" in b.variables
    ):

        tc = np.asarray(
            c["time"][:]
        )

        ts = np.concatenate([
            np.asarray(
                a["time"][:]
            ),
            np.asarray(
                b["time"][:]
            ),
        ])


        time_equal = np.array_equal(
            tc,
            ts
        )


        print(
            "time coordinate:",
            "PASS"
            if time_equal
            else "FAIL"
        )

        print(
            "  continuous =",
            tc
        )

        print(
            "  segmented  =",
            ts
        )

    else:

        time_equal = True

        print(
            "time coordinate not present; "
            "skipping explicit time comparison"
        )


    print()


    outvars = sorted(
        name
        for name in c.variables
        if name.startswith("OUT_")
    )


    for name in outvars:

        if (
            name not in a.variables or
            name not in b.variables
        ):

            raise RuntimeError(
                f"{name} missing from segmented output."
            )


        reference = np.ma.asarray(
            c[name][:]
        )


        segmented = np.ma.concatenate([
            np.ma.asarray(
                a[name][:]
            ),
            np.ma.asarray(
                b[name][:]
            ),
        ], axis=0)


        equal = exact_equal(
            reference,
            segmented
        )

        absmax = absmax_difference(
            reference,
            segmented
        )


        comparison.append({
            "group":
                "output",

            "variable":
                name,

            "bitwise_identical":
                equal,

            "absmax_difference":
                absmax,
        })


        print(
            f"{name:28s}",
            f"exact={str(equal):5s}",
            f"absmax={absmax:.17g}"
        )


# ============================================================
# 5. Compare complete final VIC state
# ============================================================

print()
print("=" * 72)
print("H5b FINAL-STATE EQUIVALENCE")
print("=" * 72)


with nc.Dataset(
    cont_state
) as c, \
     nc.Dataset(
         seg2_state
     ) as s:


    cvars = set(
        c.variables
    )

    svars = set(
        s.variables
    )


    only_cont = sorted(
        cvars -
        svars
    )

    only_seg = sorted(
        svars -
        cvars
    )


    if only_cont:

        print(
            "Only continuous:",
            only_cont
        )


    if only_seg:

        print(
            "Only segmented:",
            only_seg
        )


    for name in sorted(
        cvars &
        svars
    ):

        x = np.ma.asarray(
            c[name][:]
        )

        y = np.ma.asarray(
            s[name][:]
        )


        equal = exact_equal(
            x,
            y
        )

        absmax = absmax_difference(
            x,
            y
        )


        comparison.append({
            "group":
                "final_state",

            "variable":
                name,

            "bitwise_identical":
                equal,

            "absmax_difference":
                absmax,
        })


        if not equal:

            print(
                f"{name:32s}",
                f"exact={equal}",
                f"absmax={absmax}"
            )


# ============================================================
# Save comparison
# ============================================================

comparison_file = (
    OUT /
    "comparison.csv"
)


with comparison_file.open(
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
# Key scientific checks
# ============================================================

lookup = {
    (
        row["group"],
        row["variable"]
    ):
        row

    for row in comparison
}


keyvars = [
    "OUT_GW_EXCHANGE",
    "OUT_SOIL_MOIST",
    "OUT_RUNOFF",
    "OUT_WATER_ERROR",
]


print()
print("=" * 72)
print("H5b KEY TESTS")
print("=" * 72)


key_pass = True


for name in keyvars:

    row = lookup.get(
        (
            "output",
            name,
        )
    )


    if row is None:

        print(
            name,
            "MISSING"
        )

        key_pass = False
        continue


    passed = bool(
        row[
            "bitwise_identical"
        ]
    )

    key_pass &= passed


    print(
        f"{name:24s}",
        "PASS"
        if passed
        else "FAIL",
        "absmax =",
        row[
            "absmax_difference"
        ],
    )


all_output_pass = all(
    row[
        "bitwise_identical"
    ]
    for row in comparison
    if row["group"] == "output"
)


all_state_pass = (
    not only_cont and
    not only_seg and
    all(
        row[
            "bitwise_identical"
        ]
        for row in comparison
        if row["group"] ==
           "final_state"
    )
)


overall = (
    time_equal and
    key_pass and
    all_output_pass and
    all_state_pass
)


print()

print(
    "TIME COORDINATE:",
    "PASS"
    if time_equal
    else "FAIL"
)

print(
    "ALL OUTPUT VARIABLES:",
    "PASS"
    if all_output_pass
    else "FAIL"
)

print(
    "FINAL VIC STATE:",
    "PASS"
    if all_state_pass
    else "FAIL"
)


print()
print("=" * 72)

print(
    "H5b 12-HOUR RESTART EQUIVALENCE:",
    "PASS"
    if overall
    else "REQUIRES REVIEW"
)

print("=" * 72)


print()
print("Continuous output:")
print(cont_flux)

print()
print("Half 1 output:")
print(seg1_flux)

print()
print("Half 1 state:")
print(seg1_state)

print()
print("Half 2 output:")
print(seg2_flux)

print()
print("Final segmented state:")
print(seg2_state)

print()
print("Comparison:")
print(comparison_file)

print()
print("Experiment directory:")
print(OUT)
