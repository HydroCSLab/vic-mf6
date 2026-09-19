#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
from datetime import datetime, timedelta
import csv
import math
import os
import shutil
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

DOMAIN_FILE = (
    PARAMDIR /
    "domain.stehekin.20151028.nc"
)

LIBMF6 = Path(
    str(MF6_LIBRARY)
)

EXPROOT = Path(
    str(RESULT_ROOT)
)

SS_PER_M = float(
    os.environ.get(
        "H5C_SS_PER_M",
        "0.001"
    )
)

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    EXPROOT /
    f"H5c_subdaily_coupling_SS_{SS_PER_M:.0e}_{STAMP}"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experiment definition
# ============================================================

START = datetime(
    1949, 1, 1, 0, 0, 0
)

TOTAL_HOURS = 240

INTERVAL_HOURS = [
    int(x)
    for x in os.environ.get(
        "H5C_INTERVAL_HOURS",
        "24,12,6,3,1"
    ).split(",")
]

INITIAL_HEAD_M = -105.328

EXCHANGE_LENGTH_M = 100.0
KA_SCALE = 0.001
DRAIN_FRACTION = 1.0

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)



# ============================================================
# VIC domain areas
# ============================================================

with nc.Dataset(DOMAIN_FILE) as ds:

    area_var = ds["area"]

    area = np.ma.asarray(
        area_var[:],
        dtype=float,
    )

    domain_mask = (
        np.asarray(
            ds["mask"][:]
        ) > 0
    )

    units = (
        getattr(
            area_var,
            "units",
            ""
        )
        .strip()
        .lower()
        .replace(" ", "")
    )


if units in (
    "m2",
    "m^2",
    "m**2",
):

    area_m2 = area

elif units in (
    "km2",
    "km^2",
    "km**2",
):

    area_m2 = (
        area *
        1.0e6
    )

else:

    raise RuntimeError(
        f"Unsupported area units: {units!r}"
    )


valid_area = (
    domain_mask &
    ~np.ma.getmaskarray(
        area_m2
    )
)


TOTAL_AREA_M2 = float(
    np.sum(
        np.asarray(area_m2)[
            valid_area
        ]
    )
)


CELL_WIDTH_M = math.sqrt(
    TOTAL_AREA_M2
)


STORAGE_COEFF_M2 = (
    SS_PER_M *
    TOTAL_AREA_M2 *
    THICKNESS_M
)


print()
print("===== H5c DOMAIN =====")

print(
    "active VIC cells   =",
    int(np.sum(valid_area))
)

print(
    "total VIC area     =",
    TOTAL_AREA_M2,
    "m2"
)

print(
    "MF6 storage coeff  =",
    STORAGE_COEFF_M2,
    "m2"
)


# ============================================================
# Provenance
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

shutil.copy2(
    BASE_GLOBAL,
    OUT / "canonical_global.txt"
)


# ============================================================
# Clean canonical global
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


clean_lines = []

for line in BASE_GLOBAL.read_text().splitlines():

    s = line.strip()

    if (
        s and
        not s.startswith("#")
    ):

        key = s.split()[0]

        if key in controlled:
            continue

    clean_lines.append(line)


BASE_TEXT = (
    "\n".join(clean_lines) +
    "\n"
)


# ============================================================
# Time helpers
# ============================================================

def seconds_of_day(t):

    return (
        t.hour * 3600 +
        t.minute * 60 +
        t.second
    )


# ============================================================
# VIC global generator
# ============================================================

def make_global(
    path,
    start_time,
    nrecs,
    interval_hours,
    result_dir,
    state_prefix,
    state_time,
    init_state,
):

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    state_prefix.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    lines = [
        "",
        "# H5c subdaily closed coupling",

        f"STARTYEAR    {start_time.year}",
        f"STARTMONTH   {start_time.month}",
        f"STARTDAY     {start_time.day}",
        f"STARTSEC     {seconds_of_day(start_time)}",

        f"NRECS        {nrecs}",

        f"RESULT_DIR   {result_dir}",

        # One output record for this entire coupling chunk.
        f"AGGFREQ      NHOURS {interval_hours}",
    ]


    if init_state is not None:

        lines.append(
            f"INIT_STATE   {init_state}"
        )


    lines.extend([
        f"STATENAME    {state_prefix}",

        f"STATEYEAR    {state_time.year}",
        f"STATEMONTH   {state_time.month}",
        f"STATEDAY     {state_time.day}",
        f"STATESEC     {seconds_of_day(state_time)}",

        "STATE_FORMAT NETCDF4_CLASSIC",
        "",
    ])


    path.write_text(
        BASE_TEXT +
        "\n".join(lines)
    )


# ============================================================
# File helper
# ============================================================

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
# Read one VIC coupling chunk
# ============================================================

def read_vic_chunk(
    flux_file,
):

    with nc.Dataset(
        flux_file
    ) as ds:

        q = np.ma.asarray(
            ds[
                "OUT_GW_EXCHANGE"
            ][:]
        )

        soil = np.ma.asarray(
            ds[
                "OUT_SOIL_MOIST"
            ][:]
        )

        runoff = np.ma.asarray(
            ds[
                "OUT_RUNOFF"
            ][:]
        )

        water_error = np.ma.asarray(
            ds[
                "OUT_WATER_ERROR"
            ][:]
        )


    if q.shape[0] != 1:

        raise RuntimeError(
            f"Expected one aggregated output record; "
            f"found {q.shape[0]} in {flux_file}"
        )


    qt = q[0]

    valid = (
        valid_area &
        ~np.ma.getmaskarray(qt)
    )


    av = np.asarray(
        area_m2
    )[valid]

    qv = np.asarray(
        qt
    )[valid]


    # OUT_GW_EXCHANGE is an accumulated depth over the
    # coupling interval [mm].
    volume_m3 = float(
        np.sum(
            qv *
            1.0e-3 *
            av
        )
    )


    depth_aw_mm = (
        volume_m3 /
        TOTAL_AREA_M2 *
        1000.0
    )


    bottom = (
        soil[0, -1]
    )

    valid_bottom = (
        valid_area &
        ~np.ma.getmaskarray(
            bottom
        )
    )

    ab = np.asarray(
        area_m2
    )[valid_bottom]

    bv = np.asarray(
        bottom
    )[valid_bottom]


    bottom_aw_mm = float(
        np.sum(
            bv * ab
        ) /
        np.sum(ab)
    )


    runoff_field = (
        runoff[0]
    )

    valid_runoff = (
        valid_area &
        ~np.ma.getmaskarray(
            runoff_field
        )
    )

    ar = np.asarray(
        area_m2
    )[valid_runoff]

    rv = np.asarray(
        runoff_field
    )[valid_runoff]


    runoff_aw_mm = float(
        np.sum(
            rv * ar
        ) /
        np.sum(ar)
    )


    wb_absmax = float(
        np.max(
            np.abs(
                water_error.compressed()
            )
        )
    )


    return {
        "volume_m3":
            volume_m3,

        "area_weighted_exchange_mm":
            depth_aw_mm,

        "bottom_moist_mm":
            bottom_aw_mm,

        "runoff_mm":
            runoff_aw_mm,

        "water_error_absmax_mm":
            wb_absmax,

        "q_min_mm":
            float(
                np.min(
                    q.compressed()
                )
            ),

        "q_max_mm":
            float(
                np.max(
                    q.compressed()
                )
            ),
    }


# ============================================================
# Write matching MF6 model
# ============================================================

def write_mf6(
    work,
    interval_hours,
):

    nsteps = (
        TOTAL_HOURS //
        interval_hours
    )

    dt_days = (
        interval_hours /
        24.0
    )


    (work / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h5c.tdis
END TIMING

BEGIN MODELS
  GWF6 h5c.nam H5C
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h5c.ims H5C
END SOLUTIONGROUP
"""
    )


    (work / "h5c.tdis").write_text(
f"""BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  10.0 {nsteps} 1.0
END PERIODDATA
"""
    )


    (work / "h5c.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
    )


    (work / "h5c.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DIS6 h5c.dis DIS
  IC6  h5c.ic  IC
  NPF6 h5c.npf NPF
  STO6 h5c.sto STO
  API6 h5c.api VICAPI
  OC6  h5c.oc  OC
END PACKAGES
"""
    )


    (work / "h5c.dis").write_text(
f"""BEGIN OPTIONS
  LENGTH_UNITS METERS
END OPTIONS

BEGIN DIMENSIONS
  NLAY 1
  NROW 1
  NCOL 1
END DIMENSIONS

BEGIN GRIDDATA
  DELR
    CONSTANT {CELL_WIDTH_M:.17g}
  DELC
    CONSTANT {CELL_WIDTH_M:.17g}
  TOP
    CONSTANT {TOP_M:.17g}
  BOTM
    CONSTANT {BOT_M:.17g}
END GRIDDATA
"""
    )


    (work / "h5c.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    CONSTANT {INITIAL_HEAD_M:.17g}
END GRIDDATA
"""
    )


    (work / "h5c.npf").write_text(
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


    (work / "h5c.sto").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICONVERT
    CONSTANT 0
  SS
    CONSTANT {SS_PER_M:.17g}
  SY
    CONSTANT 0.20
END GRIDDATA

BEGIN PERIOD 1
  TRANSIENT
END PERIOD
"""
    )


    (work / "h5c.api").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND 1
END DIMENSIONS
"""
    )


    (work / "h5c.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h5c.hds
  BUDGET FILEOUT h5c.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
END PERIOD
"""
    )


    return dt_days


# ============================================================
# Run one coupling interval
# ============================================================

def run_case(
    interval_hours,
):

    tag = (
        f"dt_{interval_hours:02d}h"
    )

    root = (
        OUT /
        tag
    )

    VIC_DIR = (
        root /
        "vic"
    )

    MF6_DIR = (
        root /
        "mf6"
    )

    GLOBAL_DIR = (
        root /
        "globals"
    )

    LOG_DIR = (
        root /
        "logs"
    )


    for p in (
        VIC_DIR,
        MF6_DIR,
        GLOBAL_DIR,
        LOG_DIR,
    ):

        p.mkdir(
            parents=True,
            exist_ok=True
        )


    dt_days = write_mf6(
        MF6_DIR,
        interval_hours
    )


    nexchanges = (
        TOTAL_HOURS //
        interval_hours
    )


    print()
    print("#" * 78)

    print(
        "H5c:",
        interval_hours,
        "hour coupling"
    )

    print(
        "coupling exchanges =",
        nexchanges
    )

    print("#" * 78)


    mf6 = XmiWrapper(
        str(LIBMF6),
        working_directory=
            str(MF6_DIR),
    )

    mf6.initialize_mpi(MPI.COMM_SELF.py2f())


    rows = []

    init_state = None

    cumulative_vic_volume = 0.0

    final_bottom = np.nan
    final_runoff = np.nan
    max_wb_error = 0.0


    try:

        head = mf6.get_value_ptr(
            mf6.get_var_address(
                "X",
                "H5C"
            )
        )

        nbound = mf6.get_value_ptr(
            mf6.get_var_address(
                "NBOUND",
                "H5C",
                "VICAPI"
            )
        )

        nodelist = mf6.get_value_ptr(
            mf6.get_var_address(
                "NODELIST",
                "H5C",
                "VICAPI"
            )
        )

        rhs = mf6.get_value_ptr(
            mf6.get_var_address(
                "RHS",
                "H5C",
                "VICAPI"
            )
        )

        hcof = mf6.get_value_ptr(
            mf6.get_var_address(
                "HCOF",
                "H5C",
                "VICAPI"
            )
        )

        simvals = mf6.get_value_ptr(
            mf6.get_var_address(
                "SIMVALS",
                "H5C",
                "VICAPI"
            )
        )


        for k in range(
            nexchanges
        ):

            chunk_start = (
                START +
                timedelta(
                    hours=
                        k *
                        interval_hours
                )
            )

            chunk_end = (
                chunk_start +
                timedelta(
                    hours=
                        interval_hours
                )
            )


            head_before = float(
                head[0]
            )


            label = (
                f"chunk_{k + 1:03d}_"
                f"{chunk_start:%Y%m%d_%H%M%S}"
            )


            chunk_root = (
                VIC_DIR /
                label
            )

            result_dir = (
                chunk_root /
                "results"
            )

            state_dir = (
                chunk_root /
                "state"
            )


            result_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            state_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            global_file = (
                GLOBAL_DIR /
                f"{label}.global.txt"
            )


            make_global(
                path=
                    global_file,

                start_time=
                    chunk_start,

                nrecs=
                    interval_hours,

                interval_hours=
                    interval_hours,

                result_dir=
                    result_dir,

                state_prefix=
                    state_dir / "state",

                state_time=
                    chunk_end,

                init_state=
                    init_state,
            )


            env = os.environ.copy()

            env.update({
                "OMP_NUM_THREADS":
                    "1",

                "VIC_ALLOW_PARTIAL_DAY":
                    "1",

                "VIC_GW_FORMULATION":
                    "head",

                "VIC_GW_REFERENCE_DEPTH":
                    "base",

                "VIC_GW_TEST_GAP_M":
                    "105.328",

                "VIC_GW_TEST_KA_SCALE":
                    str(KA_SCALE),

                "VIC_GW_MAX_DRAIN_FRACTION":
                    str(
                        DRAIN_FRACTION
                    ),

                "VIC_GW_HEAD_OFFSET_M":
                    f"{head_before:.17g}",

                "VIC_GW_EXCHANGE_LENGTH_M":
                    str(
                        EXCHANGE_LENGTH_M
                    ),
            })


            stdout = (
                LOG_DIR /
                f"{label}.stdout.log"
            )

            stderr = (
                LOG_DIR /
                f"{label}.stderr.log"
            )


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
                    env=env,
                    stdout=out,
                    stderr=err,
                )


            if proc.returncode != 0:

                raise RuntimeError(
                    f"VIC failed for {tag} "
                    f"chunk {k + 1}; "
                    f"see {stderr}"
                )


            flux_file = find_one(
                result_dir,
                "fluxes*.nc"
            )

            state_file = find_one(
                state_dir,
                "state*.nc"
            )


            info = read_vic_chunk(
                flux_file
            )


            volume_m3 = (
                info[
                    "volume_m3"
                ]
            )


            # Equivalent constant MF6 volumetric rate for
            # this coupling interval.
            Q_m3_day = (
                volume_m3 /
                dt_days
            )


            cumulative_vic_volume += (
                volume_m3
            )


            final_bottom = (
                info[
                    "bottom_moist_mm"
                ]
            )

            final_runoff = (
                info[
                    "runoff_mm"
                ]
            )

            max_wb_error = max(
                max_wb_error,
                info[
                    "water_error_absmax_mm"
                ]
            )


            # =================================================
            # MF6 solve for same coupling interval
            # =================================================

            mf6.prepare_time_step(
                dt_days
            )


            nbound[0] = 1
            nodelist[0] = 1
            hcof[0] = 0.0
            rhs[0] = -Q_m3_day


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

                nbound[0] = 1
                nodelist[0] = 1
                hcof[0] = 0.0
                rhs[0] = -Q_m3_day


                converged = mf6.solve(
                    component
                )

                iterations = iteration


                if converged:
                    break


            if not converged:

                raise RuntimeError(
                    f"MF6 failed for {tag} "
                    f"chunk {k + 1}"
                )


            mf6.finalize_solve(
                component
            )

            mf6.finalize_time_step()


            head_after = float(
                head[0]
            )

            api_Q = float(
                simvals[0]
            )


            dh = (
                head_after -
                head_before
            )


            mf6_volume = (
                dh *
                STORAGE_COEFF_M2
            )


            transfer_error = (
                mf6_volume -
                volume_m3
            )


            flow_error = (
                api_Q -
                Q_m3_day
            )


            rows.append({
                "interval_hours":
                    interval_hours,

                "exchange":
                    k + 1,

                "start_time":
                    chunk_start.isoformat(),

                "end_time":
                    chunk_end.isoformat(),

                "mf6_head_before_m":
                    head_before,

                "vic_exchange_depth_mm":
                    info[
                        "area_weighted_exchange_mm"
                    ],

                "vic_volume_m3":
                    volume_m3,

                "mf6_Q_m3_day":
                    Q_m3_day,

                "mf6_simvals_m3_day":
                    api_Q,

                "api_flow_error_m3_day":
                    flow_error,

                "mf6_iterations":
                    iterations,

                "mf6_head_after_m":
                    head_after,

                "mf6_dh_m":
                    dh,

                "mf6_storage_delta_m3":
                    mf6_volume,

                "transfer_error_m3":
                    transfer_error,

                "bottom_moist_mm":
                    info[
                        "bottom_moist_mm"
                    ],

                "runoff_mm":
                    info[
                        "runoff_mm"
                    ],

                "water_error_absmax_mm":
                    info[
                        "water_error_absmax_mm"
                    ],
            })


            if not np.isclose(
                api_Q,
                Q_m3_day,
                rtol=1.0e-12,
                atol=1.0e-6,
            ):

                raise RuntimeError(
                    f"API-flow mismatch for "
                    f"{tag} exchange {k + 1}"
                )


            if not np.isclose(
                mf6_volume,
                volume_m3,
                rtol=1.0e-9,
                atol=1.0e-3,
            ):

                raise RuntimeError(
                    f"Volume mismatch for "
                    f"{tag} exchange {k + 1}"
                )


            init_state = (
                state_file
            )


            # Avoid printing 240 giant blocks in the 1-hour case.
            if (
                k == 0 or
                k == nexchanges - 1 or
                (k + 1) % max(
                    1,
                    nexchanges // 10
                ) == 0
            ):

                print(
                    f"[{k + 1:3d}/"
                    f"{nexchanges:3d}] "
                    f"h={head_after:.9f} "
                    f"V={volume_m3:.3f} m3"
                )


        final_head = float(
            head[0]
        )


    finally:

        mf6.finalize()


    # ========================================================
    # Conservation
    # ========================================================

    mf6_storage_change = (
        (
            final_head -
            INITIAL_HEAD_M
        ) *
        STORAGE_COEFF_M2
    )


    conservation_error = (
        mf6_storage_change -
        cumulative_vic_volume
    )


    chunk_file = (
        root /
        "coupling_steps.csv"
    )


    with chunk_file.open(
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


    return {
        "interval_hours":
            interval_hours,

        "n_exchanges":
            nexchanges,

        "final_mf6_head_m":
            final_head,

        "mf6_head_change_m":
            final_head -
            INITIAL_HEAD_M,

        "cumulative_vic_volume_m3":
            cumulative_vic_volume,

        "cumulative_exchange_mm":
            cumulative_vic_volume /
            TOTAL_AREA_M2 *
            1000.0,

        "mf6_storage_change_m3":
            mf6_storage_change,

        "conservation_error_m3":
            conservation_error,

        "relative_conservation_error":
            (
                conservation_error /
                cumulative_vic_volume
                if cumulative_vic_volume != 0.0
                else 0.0
            ),

        "final_bottom_moist_mm":
            final_bottom,

        "final_runoff_mm":
            final_runoff,

        "max_vic_water_error_mm":
            max_wb_error,

        "max_transfer_error_m3":
            max(
                abs(
                    row[
                        "transfer_error_m3"
                    ]
                )
                for row in rows
            ),

        "max_api_flow_error_m3_day":
            max(
                abs(
                    row[
                        "api_flow_error_m3_day"
                    ]
                )
                for row in rows
            ),
    }


# ============================================================
# Run all intervals
# ============================================================

results = []


for interval in INTERVAL_HOURS:

    results.append(
        run_case(
            interval
        )
    )


# ============================================================
# 1-hour reference
# ============================================================

reference = next(
    row
    for row in results
    if row[
        "interval_hours"
    ] == 1
)


for row in results:

    row[
        "final_head_diff_vs_1h_m"
    ] = (
        row[
            "final_mf6_head_m"
        ] -
        reference[
            "final_mf6_head_m"
        ]
    )


    row[
        "volume_diff_vs_1h_m3"
    ] = (
        row[
            "cumulative_vic_volume_m3"
        ] -
        reference[
            "cumulative_vic_volume_m3"
        ]
    )


    row[
        "volume_diff_vs_1h_percent"
    ] = (
        100.0 *
        row[
            "volume_diff_vs_1h_m3"
        ] /
        abs(
            reference[
                "cumulative_vic_volume_m3"
            ]
        )
    )


    row[
        "bottom_moist_diff_vs_1h_mm"
    ] = (
        row[
            "final_bottom_moist_mm"
        ] -
        reference[
            "final_bottom_moist_mm"
        ]
    )


# ============================================================
# H3b 24-hour regression
# ============================================================

h3b_dirs = sorted(
    EXPROOT.glob(
        "H3b_closed_loop_*"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)


h3b_regression_pass = None


if h3b_dirs:

    h3b_summary = (
        h3b_dirs[0] /
        "summary.csv"
    )

    if h3b_summary.exists():

        with h3b_summary.open() as f:

            old = next(
                csv.DictReader(f)
            )


        old_head = float(
            old[
                "final_mf6_head_m"
            ]
        )

        old_volume = float(
            old[
                "cumulative_vic_transfer_m3"
            ]
        )


        day_case = next(
            row
            for row in results
            if row[
                "interval_hours"
            ] == 24
        )


        h3b_regression_pass = (
            np.isclose(
                day_case[
                    "final_mf6_head_m"
                ],
                old_head,
                rtol=0.0,
                atol=1.0e-12,
            )
            and
            np.isclose(
                day_case[
                    "cumulative_vic_volume_m3"
                ],
                old_volume,
                rtol=0.0,
                atol=1.0e-5,
            )
        )


# ============================================================
# Save summary
# ============================================================

summary_file = (
    OUT /
    "summary.csv"
)


with summary_file.open(
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


# ============================================================
# Print scientific summary
# ============================================================

print()
print("=" * 118)
print("H5c SUBDAILY COUPLING CONVERGENCE")
print("=" * 118)

print(
    f"{'dt(h)':>6s} "
    f"{'n':>5s} "
    f"{'final head':>16s} "
    f"{'cum volume':>18s} "
    f"{'dV vs 1h':>14s} "
    f"{'% vs 1h':>12s} "
    f"{'dh vs 1h':>14s} "
    f"{'dS3 vs 1h':>14s} "
    f"{'cons err':>13s}"
)


for row in results:

    print(
        f"{row['interval_hours']:6d} "
        f"{row['n_exchanges']:5d} "
        f"{row['final_mf6_head_m']:16.10f} "
        f"{row['cumulative_vic_volume_m3']:18.3f} "
        f"{row['volume_diff_vs_1h_m3']:14.3f} "
        f"{row['volume_diff_vs_1h_percent']:12.7f} "
        f"{row['final_head_diff_vs_1h_m']:14.7e} "
        f"{row['bottom_moist_diff_vs_1h_mm']:14.7e} "
        f"{row['conservation_error_m3']:13.3e}"
    )


print()
print(
    "1-hour reference final head =",
    reference[
        "final_mf6_head_m"
    ],
    "m"
)

print(
    "1-hour reference volume     =",
    reference[
        "cumulative_vic_volume_m3"
    ],
    "m3"
)


if h3b_regression_pass is not None:

    print()

    print(
        "24-HOUR H3b REGRESSION:",
        "PASS"
        if h3b_regression_pass
        else "FAIL"
    )


print()
print("Saved:")
print(summary_file)

print()
print("Experiment directory:")
print(OUT)
