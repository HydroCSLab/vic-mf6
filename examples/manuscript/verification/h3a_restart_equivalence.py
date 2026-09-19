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
    f"H3a_restart_equivalence_{STAMP}"
)

GLOBALS = OUT / "globals"
LOGS = OUT / "logs"
CONTINUOUS_DIR = OUT / "continuous"
SEGMENTED_DIR = OUT / "segmented"

for p in (
    GLOBALS,
    LOGS,
    CONTINUOUS_DIR,
    SEGMENTED_DIR,
):
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment definition
# ============================================================

START = date(1949, 1, 1)
NDAYS = 10

HEAD_OFFSET_M = -105.328
EXCHANGE_LENGTH_M = 100.0
KA_SCALE = 0.001
DRAIN_FRACTION = 1.0


env = os.environ.copy()

env.update({
    "OMP_NUM_THREADS": "1",

    "VIC_GW_FORMULATION":
        "head",

    "VIC_GW_REFERENCE_DEPTH":
        "base",

    "VIC_GW_TEST_GAP_M":
        "105.328",

    "VIC_GW_TEST_KA_SCALE":
        str(KA_SCALE),

    "VIC_GW_MAX_DRAIN_FRACTION":
        str(DRAIN_FRACTION),

    "VIC_GW_HEAD_OFFSET_M":
        str(HEAD_OFFSET_M),

    "VIC_GW_EXCHANGE_LENGTH_M":
        str(EXCHANGE_LENGTH_M),
})


# ============================================================
# Provenance
# ============================================================

def capture(cmd):
    from manuscript_paths import source_provenance
    return source_provenance(cmd)


(OUT / "vic_commit.txt").write_text(
    capture(["git", "rev-parse", "HEAD"])
)

(OUT / "vic_status.txt").write_text(
    capture(["git", "status", "--short"])
)

(OUT / "vic_diff.txt").write_text(
    capture(["git", "diff"])
)

shutil.copy2(
    BASE_GLOBAL,
    OUT / "canonical_global.txt"
)


(OUT / "EXPERIMENT_H3a.md").write_text(
f"""# H3a — VIC restart equivalence

Purpose
-------

Test whether daily VIC state/restart segmentation reproduces an
otherwise identical continuous 10-day simulation.

Both simulations use the same fixed groundwater boundary:

- formulation: head
- hydraulic reference: soil-column base
- groundwater head offset: {HEAD_OFFSET_M} m
- exchange length: {EXCHANGE_LENGTH_M} m
- Ka scale: {KA_SCALE}
- downward limiter: {DRAIN_FRACTION}

Control
-------

One continuous run:

1949-01-01 through 1949-01-10.

Segmented experiment
--------------------

Ten one-day runs.

The state written at the end of each daily interval is supplied as
INIT_STATE to the following interval.

The final state timestamp is 1949-01-11 00:00:00.

Success criteria
----------------

1. Daily output fields from the segmented run reproduce the continuous
   simulation.
2. OUT_GW_EXCHANGE is identical.
3. OUT_SOIL_MOIST is identical.
4. OUT_RUNOFF is identical.
5. OUT_WATER_ERROR is identical.
6. Final VIC state variables reproduce the continuous-run final state.
"""
)


# ============================================================
# Build temporary global parameter files
# ============================================================

base_text = BASE_GLOBAL.read_text()

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
}


def strip_controlled(text):

    out = []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:
            out.append(line)
            continue

        if stripped.startswith("#"):
            out.append(line)
            continue

        key = stripped.split()[0]

        if key in controlled:
            continue

        out.append(line)

    return "\n".join(out) + "\n"


clean_global = strip_controlled(
    base_text
)


def make_global(
    filename,
    start_day,
    end_day,
    result_dir,
    state_prefix,
    state_time,
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

    block = [
        "",
        "# ================================================",
        "# H3a restart-equivalence controls",
        "# ================================================",

        f"STARTYEAR    {start_day.year}",
        f"STARTMONTH   {start_day.month}",
        f"STARTDAY     {start_day.day}",
        "STARTSEC     0",

        f"ENDYEAR      {end_day.year}",
        f"ENDMONTH     {end_day.month}",
        f"ENDDAY       {end_day.day}",

        f"RESULT_DIR   {result_dir}",
    ]

    if init_state is not None:
        block.append(
            f"INIT_STATE   {init_state}"
        )

    block.extend([
        f"STATENAME    {state_prefix}",
        f"STATEYEAR    {state_time.year}",
        f"STATEMONTH   {state_time.month}",
        f"STATEDAY     {state_time.day}",
        "STATESEC     0",
        "STATE_FORMAT NETCDF4_CLASSIC",
        "",
    ])

    filename.write_text(
        clean_global +
        "\n".join(block)
    )


# ============================================================
# VIC execution helper
# ============================================================

def run_vic(
    label,
    global_file,
):

    stdout_file = (
        LOGS /
        f"{label}.stdout.log"
    )

    stderr_file = (
        LOGS /
        f"{label}.stderr.log"
    )

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    with stdout_file.open("w") as out, \
         stderr_file.open("w") as err:

        result = subprocess.run(
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
            stdout=out,
            stderr=err,
        )

    if result.returncode != 0:

        print(
            "[FAILED]",
            label
        )

        print(
            "stderr:",
            stderr_file
        )

        raise RuntimeError(
            f"VIC failed: {label}"
        )

    print(
        "[OK]",
        label
    )


def one_file(directory, pattern):

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
# A. Continuous 10-day reference
# ============================================================

continuous_results = (
    CONTINUOUS_DIR /
    "results"
)

continuous_state_dir = (
    CONTINUOUS_DIR /
    "state"
)

continuous_state_prefix = (
    continuous_state_dir /
    "state"
)

continuous_global = (
    GLOBALS /
    "continuous.global.txt"
)

final_timestamp = (
    START +
    timedelta(days=NDAYS)
)


make_global(
    filename=
        continuous_global,

    start_day=
        START,

    end_day=
        START +
        timedelta(days=NDAYS - 1),

    result_dir=
        continuous_results,

    state_prefix=
        continuous_state_prefix,

    state_time=
        final_timestamp,
)


run_vic(
    "continuous",
    continuous_global
)


continuous_flux = one_file(
    continuous_results,
    "fluxes*.nc"
)

continuous_state = one_file(
    continuous_state_dir,
    "state*.nc"
)


print()
print(
    "Continuous output:",
    continuous_flux
)

print(
    "Continuous final state:",
    continuous_state
)


# ============================================================
# B. Ten one-day restart segments
# ============================================================

daily_flux_files = []

init_state = None

segmented_final_state = None


for n in range(NDAYS):

    current_day = (
        START +
        timedelta(days=n)
    )

    next_day = (
        current_day +
        timedelta(days=1)
    )

    label = (
        f"day_{n + 1:02d}_"
        f"{current_day:%Y%m%d}"
    )

    daydir = (
        SEGMENTED_DIR /
        label
    )

    result_dir = (
        daydir /
        "results"
    )

    state_dir = (
        daydir /
        "state"
    )

    state_prefix = (
        state_dir /
        "state"
    )

    global_file = (
        GLOBALS /
        f"{label}.global.txt"
    )


    make_global(
        filename=
            global_file,

        start_day=
            current_day,

        end_day=
            current_day,

        result_dir=
            result_dir,

        state_prefix=
            state_prefix,

        state_time=
            next_day,

        init_state=
            init_state,
    )


    run_vic(
        label,
        global_file
    )


    flux_file = one_file(
        result_dir,
        "fluxes*.nc"
    )

    state_file = one_file(
        state_dir,
        "state*.nc"
    )


    daily_flux_files.append(
        flux_file
    )

    init_state = state_file

    segmented_final_state = (
        state_file
    )


    print(
        "  flux :",
        flux_file.name
    )

    print(
        "  state:",
        state_file.name
    )


# ============================================================
# Comparison helpers
# ============================================================

def exact_equal(a, b):

    a = np.ma.asarray(a)
    b = np.ma.asarray(b)

    if a.shape != b.shape:
        return False

    ma = np.ma.getmaskarray(a)
    mb = np.ma.getmaskarray(b)

    if not np.array_equal(
        ma,
        mb
    ):
        return False

    av = a.compressed()
    bv = b.compressed()

    if av.dtype.kind in "fc":

        return np.array_equal(
            av,
            bv,
            equal_nan=True,
        )

    return np.array_equal(
        av,
        bv
    )


def absmax_difference(a, b):

    a = np.ma.asarray(a)
    b = np.ma.asarray(b)

    if a.shape != b.shape:
        return np.nan

    mask = (
        np.ma.getmaskarray(a) |
        np.ma.getmaskarray(b)
    )

    av = np.asarray(a)
    bv = np.asarray(b)

    good = ~mask

    if not np.any(good):
        return 0.0

    if (
        av.dtype.kind not in "fciu"
        or bv.dtype.kind not in "fciu"
    ):
        return np.nan

    return float(
        np.max(
            np.abs(
                av[good] -
                bv[good]
            )
        )
    )


# ============================================================
# C. Compare output trajectories
# ============================================================

print()
print("=" * 72)
print("H3a OUTPUT EQUIVALENCE")
print("=" * 72)


comparison_rows = []


with nc.Dataset(
    continuous_flux
) as cds:

    outvars = sorted(
        name
        for name in cds.variables
        if name.startswith("OUT_")
    )


    for name in outvars:

        reference = np.ma.asarray(
            cds[name][:]
        )

        pieces = []

        for f in daily_flux_files:

            with nc.Dataset(f) as ds:

                if name not in ds.variables:

                    raise RuntimeError(
                        f"{name} missing from {f}"
                    )

                pieces.append(
                    np.ma.asarray(
                        ds[name][:]
                    )
                )


        segmented = np.ma.concatenate(
            pieces,
            axis=0
        )


        equal = exact_equal(
            reference,
            segmented
        )

        absmax = absmax_difference(
            reference,
            segmented
        )


        comparison_rows.append({
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
# D. Compare final VIC states
# ============================================================

print()
print("=" * 72)
print("H3a FINAL-STATE EQUIVALENCE")
print("=" * 72)


with nc.Dataset(
    continuous_state
) as a, \
     nc.Dataset(
         segmented_final_state
     ) as b:

    common = sorted(
        set(a.variables) &
        set(b.variables)
    )

    only_a = sorted(
        set(a.variables) -
        set(b.variables)
    )

    only_b = sorted(
        set(b.variables) -
        set(a.variables)
    )


    if only_a:
        print(
            "Only continuous:",
            only_a
        )

    if only_b:
        print(
            "Only segmented:",
            only_b
        )


    for name in common:

        x = np.ma.asarray(
            a[name][:]
        )

        y = np.ma.asarray(
            b[name][:]
        )

        equal = exact_equal(
            x,
            y
        )

        absmax = absmax_difference(
            x,
            y
        )


        comparison_rows.append({
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
# Save comparison table
# ============================================================

summary_file = (
    OUT /
    "comparison.csv"
)

with summary_file.open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            comparison_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        comparison_rows
    )


# ============================================================
# Key scientific tests
# ============================================================

key_variables = [
    "OUT_GW_EXCHANGE",
    "OUT_SOIL_MOIST",
    "OUT_RUNOFF",
    "OUT_WATER_ERROR",
]


lookup = {
    (
        row["group"],
        row["variable"]
    ):
        row

    for row in comparison_rows
}


print()
print("=" * 72)
print("H3a KEY TESTS")
print("=" * 72)


key_pass = True

for name in key_variables:

    row = lookup.get(
        ("output", name)
    )

    if row is None:

        print(
            name,
            "MISSING"
        )

        key_pass = False
        continue

    passed = bool(
        row["bitwise_identical"]
    )

    key_pass &= passed

    print(
        f"{name:24s}",
        "PASS"
        if passed
        else "FAIL",
        "absmax =",
        row["absmax_difference"],
    )


all_outputs_exact = all(
    row["bitwise_identical"]
    for row in comparison_rows
    if row["group"] == "output"
)


all_state_exact = (
    not only_a
    and not only_b
    and all(
        row["bitwise_identical"]
        for row in comparison_rows
        if row["group"] ==
           "final_state"
    )
)


print()
print(
    "ALL OUTPUT VARIABLES:",
    "PASS"
    if all_outputs_exact
    else "FAIL"
)

print(
    "FINAL VIC STATE:",
    "PASS"
    if all_state_exact
    else "FAIL"
)


overall = (
    key_pass
    and all_outputs_exact
    and all_state_exact
)


print()
print("=" * 72)

if overall:

    print(
        "H3a RESTART EQUIVALENCE: PASS"
    )

else:

    print(
        "H3a RESTART EQUIVALENCE: "
        "REQUIRES REVIEW"
    )

print("=" * 72)

print()
print("Experiment directory:")
print(OUT)

print()
print("Comparison table:")
print(summary_file)
