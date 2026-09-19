#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
from datetime import date, timedelta
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

STAMP = time.strftime("%Y%m%d_%H%M%S")

OUT = (
    EXPROOT /
    f"H3b_closed_loop_{STAMP}"
)

VIC_OUT = OUT / "vic"
GLOBALS = OUT / "globals"
LOGS = OUT / "logs"
MF6_WORK = OUT / "mf6"

for p in (
    OUT,
    VIC_OUT,
    GLOBALS,
    LOGS,
    MF6_WORK,
):
    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Coupling configuration
# ============================================================

START = date(1949, 1, 1)
NDAYS = 10

INITIAL_MF6_HEAD_M = -105.328

EXCHANGE_LENGTH_M = 100.0
KA_SCALE = 0.001
DRAIN_FRACTION = 1.0

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = (
    TOP_M -
    BOT_M
)

SS_PER_M = 0.001

DT_DAY = 1.0


# ============================================================
# Read VIC active-domain area
# ============================================================

with nc.Dataset(
    DOMAIN_FILE
) as ds:

    area_var = ds["area"]

    area = np.ma.asarray(
        area_var[:]
    )

    mask = (
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

    area_m2 = np.ma.asarray(
        area,
        dtype=float
    )

elif units in (
    "km2",
    "km^2",
    "km**2",
):

    area_m2 = (
        np.ma.asarray(
            area,
            dtype=float
        ) *
        1.0e6
    )

else:

    raise RuntimeError(
        f"Unsupported domain area units: {units!r}"
    )


valid_area = (
    mask &
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
print("===== H3b DOMAIN =====")

print(
    "active cells        =",
    int(
        np.sum(
            valid_area
        )
    )
)

print(
    "total area          =",
    TOTAL_AREA_M2,
    "m2"
)

print(
    "MF6 cell width      =",
    CELL_WIDTH_M,
    "m"
)

print(
    "MF6 storage coeff   =",
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
# Create persistent 10-day MF6 model
# ============================================================

(MF6_WORK / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h3b.tdis
END TIMING

BEGIN MODELS
  GWF6 h3b.nam H3B
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h3b.ims H3B
END SOLUTIONGROUP
"""
)


(MF6_WORK / "h3b.tdis").write_text(
f"""BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  {float(NDAYS):.15g} {NDAYS} 1.0
END PERIODDATA
"""
)


(MF6_WORK / "h3b.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
)


(MF6_WORK / "h3b.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DIS6 h3b.dis DIS
  IC6  h3b.ic  IC
  NPF6 h3b.npf NPF
  STO6 h3b.sto STO
  API6 h3b.api VICAPI
  OC6  h3b.oc  OC
END PACKAGES
"""
)


(MF6_WORK / "h3b.dis").write_text(
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
    CONSTANT {CELL_WIDTH_M:.15g}
  DELC
    CONSTANT {CELL_WIDTH_M:.15g}
  TOP
    CONSTANT {TOP_M:.15g}
  BOTM
    CONSTANT {BOT_M:.15g}
END GRIDDATA
"""
)


(MF6_WORK / "h3b.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    CONSTANT {INITIAL_MF6_HEAD_M:.15g}
END GRIDDATA
"""
)


(MF6_WORK / "h3b.npf").write_text(
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


(MF6_WORK / "h3b.sto").write_text(
f"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN GRIDDATA
  ICONVERT
    CONSTANT 0
  SS
    CONSTANT {SS_PER_M:.15g}
  SY
    CONSTANT 0.20
END GRIDDATA

BEGIN PERIOD 1
  TRANSIENT
END PERIOD
"""
)


(MF6_WORK / "h3b.api").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND 1
END DIMENSIONS
"""
)


(MF6_WORK / "h3b.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h3b.hds
  BUDGET FILEOUT h3b.cbc
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD ALL
  SAVE BUDGET ALL
  PRINT HEAD LAST
  PRINT BUDGET LAST
END PERIOD
"""
)


# ============================================================
# Create VIC global-file template
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


clean_global = (
    "\n".join(lines) +
    "\n"
)


def make_global(
    path,
    current_day,
    next_day,
    result_dir,
    state_prefix,
    init_state,
):

    block = [
        "",
        "# ================================================",
        "# H3b closed-loop daily coupling",
        "# ================================================",

        f"STARTYEAR    {current_day.year}",
        f"STARTMONTH   {current_day.month}",
        f"STARTDAY     {current_day.day}",
        "STARTSEC     0",

        f"ENDYEAR      {current_day.year}",
        f"ENDMONTH     {current_day.month}",
        f"ENDDAY       {current_day.day}",

        f"RESULT_DIR   {result_dir}",
    ]

    if init_state is not None:

        block.append(
            f"INIT_STATE   {init_state}"
        )

    block.extend([
        f"STATENAME    {state_prefix}",

        f"STATEYEAR    {next_day.year}",
        f"STATEMONTH   {next_day.month}",
        f"STATEDAY     {next_day.day}",
        "STATESEC     0",

        "STATE_FORMAT NETCDF4_CLASSIC",
        "",
    ])

    path.write_text(
        clean_global +
        "\n".join(block)
    )


def find_one(
    directory,
    pattern,
):

    files = sorted(
        directory.glob(pattern)
    )

    if len(files) != 1:

        raise RuntimeError(
            f"Expected one {pattern} in "
            f"{directory}; found {files}"
        )

    return files[0]


# ============================================================
# Initialize MF6 once and keep it alive
# ============================================================

mf6 = XmiWrapper(
    str(LIBMF6),
    working_directory=str(MF6_WORK),
)

mf6.initialize_mpi(MPI.COMM_SELF.py2f())


rows = []

init_state = None

cumulative_vic_volume = 0.0
cumulative_mf6_storage = 0.0


try:

    # --------------------------------------------------------
    # Pointers remain allocated because MAXBOUND is fixed.
    # --------------------------------------------------------

    head = mf6.get_value_ptr(
        mf6.get_var_address(
            "X",
            "H3B"
        )
    )

    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H3B",
            "VICAPI"
        )
    )

    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H3B",
            "VICAPI"
        )
    )

    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H3B",
            "VICAPI"
        )
    )

    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H3B",
            "VICAPI"
        )
    )

    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H3B",
            "VICAPI"
        )
    )


    initial_head = float(
        head[0]
    )


    print()
    print(
        "Initial MF6 head =",
        initial_head,
        "m"
    )


    # ========================================================
    # Explicit daily coupling loop
    # ========================================================

    for n in range(NDAYS):

        current_day = (
            START +
            timedelta(days=n)
        )

        next_day = (
            current_day +
            timedelta(days=1)
        )


        mf6_head_before = float(
            head[0]
        )


        print()
        print("=" * 76)

        print(
            f"COUPLING DAY {n + 1:02d} "
            f"{current_day.isoformat()}"
        )

        print("=" * 76)

        print(
            "MF6 head -> VIC =",
            mf6_head_before,
            "m"
        )


        # ----------------------------------------------------
        # VIC daily segment
        # ----------------------------------------------------

        label = (
            f"day_{n + 1:02d}_"
            f"{current_day:%Y%m%d}"
        )

        daydir = (
            VIC_OUT /
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

        for p in (
            result_dir,
            state_dir,
        ):
            p.mkdir(
                parents=True,
                exist_ok=True
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
            path=
                global_file,

            current_day=
                current_day,

            next_day=
                next_day,

            result_dir=
                result_dir,

            state_prefix=
                state_prefix,

            init_state=
                init_state,
        )


        vic_env = os.environ.copy()

        vic_env.update({
            "OMP_NUM_THREADS":
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
                str(DRAIN_FRACTION),

            # MF6 uses the fixed coupling interface as
            # elevation zero, so its head is the head offset
            # consumed by VIC.
            "VIC_GW_HEAD_OFFSET_M":
                f"{mf6_head_before:.17g}",

            "VIC_GW_EXCHANGE_LENGTH_M":
                str(EXCHANGE_LENGTH_M),
        })


        stdout_file = (
            LOGS /
            f"{label}.stdout.log"
        )

        stderr_file = (
            LOGS /
            f"{label}.stderr.log"
        )


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
                env=vic_env,
                stdout=out,
                stderr=err,
            )


        if result.returncode != 0:

            raise RuntimeError(
                f"VIC failed on {label}; "
                f"see {stderr_file}"
            )


        flux_file = find_one(
            result_dir,
            "fluxes*.nc"
        )

        state_file = find_one(
            state_dir,
            "state*.nc"
        )


        # ----------------------------------------------------
        # Read actual VIC signed groundwater exchange
        # ----------------------------------------------------

        with nc.Dataset(
            flux_file
        ) as ds:

            q = np.ma.asarray(
                ds[
                    "OUT_GW_EXCHANGE"
                ][0]
            )

            water_error = (
                np.ma.asarray(
                    ds[
                        "OUT_WATER_ERROR"
                    ][0]
                )
            )

            runoff = np.ma.asarray(
                ds[
                    "OUT_RUNOFF"
                ][0]
            )

            soil = np.ma.asarray(
                ds[
                    "OUT_SOIL_MOIST"
                ][0]
            )


        valid = (
            valid_area &
            ~np.ma.getmaskarray(q)
        )


        q_values = (
            np.asarray(q)[
                valid
            ]
        )

        a_values = (
            np.asarray(area_m2)[
                valid
            ]
        )


        # VIC daily amount [mm] -> volume [m3].
        #
        # Since coupling interval is exactly one day,
        # this has the same numerical value as m3/day
        # for the constant-rate MF6 API boundary.
        cell_volume_m3 = (
            q_values *
            1.0e-3 *
            a_values
        )

        Q_m3_day = float(
            np.sum(
                cell_volume_m3
            )
        )


        q_area_weighted = (
            Q_m3_day /
            TOTAL_AREA_M2 *
            1000.0
        )


        cumulative_vic_volume += (
            Q_m3_day *
            DT_DAY
        )


        print(
            "VIC area-weighted q =",
            q_area_weighted,
            "mm/day"
        )

        print(
            "VIC -> MF6 Q        =",
            Q_m3_day,
            "m3/day"
        )


        # ----------------------------------------------------
        # Advance one MF6 timestep with the VIC flux
        # ----------------------------------------------------

        mf6.prepare_time_step(
            DT_DAY
        )


        nbound[0] = 1
        nodelist[0] = 1

        hcof[0] = 0.0

        # MF6 package convention:
        #
        # Q_into_cell = HCOF*h - RHS
        #
        # therefore with HCOF = 0:
        #
        # RHS = -Q_VIC
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

            # Explicitly maintain boundary every nonlinear solve.
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
                f"MF6 did not converge on day {n + 1}"
            )


        mf6.finalize_solve(
            component
        )

        mf6.finalize_time_step()


        # ----------------------------------------------------
        # MF6 diagnostics
        # ----------------------------------------------------

        mf6_head_after = float(
            head[0]
        )

        api_flow = float(
            simvals[0]
        )


        dh = (
            mf6_head_after -
            mf6_head_before
        )

        mf6_storage_delta = (
            dh *
            STORAGE_COEFF_M2
        )

        expected_volume = (
            Q_m3_day *
            DT_DAY
        )

        transfer_error = (
            mf6_storage_delta -
            expected_volume
        )

        flow_error = (
            api_flow -
            Q_m3_day
        )


        cumulative_mf6_storage += (
            mf6_storage_delta
        )


        print(
            "MF6 API flow         =",
            api_flow,
            "m3/day"
        )

        print(
            "MF6 head after       =",
            mf6_head_after,
            "m"
        )

        print(
            "MF6 dh               =",
            dh,
            "m"
        )

        print(
            "transfer error       =",
            transfer_error,
            "m3"
        )


        # ----------------------------------------------------
        # VIC diagnostics
        # ----------------------------------------------------

        q_all = q.compressed()

        eps = 1.0e-10


        wb_absmax = float(
            np.max(
                np.abs(
                    water_error.compressed()
                )
            )
        )


        runoff_area_weighted_mm = float(
            np.sum(
                np.asarray(runoff)[valid] *
                a_values
            ) /
            TOTAL_AREA_M2
        )


        bottom = soil[-1]

        bottom_area_weighted_mm = float(
            np.sum(
                np.asarray(bottom)[valid] *
                a_values
            ) /
            TOTAL_AREA_M2
        )


        rows.append({
            "coupling_day":
                n + 1,

            "date":
                current_day.isoformat(),

            "mf6_head_before_m":
                mf6_head_before,

            "vic_q_area_weighted_mm_day":
                q_area_weighted,

            "vic_q_min_mm_day":
                float(
                    np.min(q_all)
                ),

            "vic_q_mean_unweighted_mm_day":
                float(
                    np.mean(q_all)
                ),

            "vic_q_max_mm_day":
                float(
                    np.max(q_all)
                ),

            "vic_upward_fraction":
                float(
                    np.mean(
                        q_all < -eps
                    )
                ),

            "vic_downward_fraction":
                float(
                    np.mean(
                        q_all > eps
                    )
                ),

            "vic_Q_m3_day":
                Q_m3_day,

            "mf6_rhs_m3_day":
                float(rhs[0]),

            "mf6_simvals_m3_day":
                api_flow,

            "api_flow_error_m3_day":
                flow_error,

            "mf6_iterations":
                iterations,

            "mf6_head_after_m":
                mf6_head_after,

            "mf6_dh_m":
                dh,

            "vic_transfer_volume_m3":
                expected_volume,

            "mf6_storage_delta_m3":
                mf6_storage_delta,

            "transfer_error_m3":
                transfer_error,

            "vic_water_error_absmax_mm":
                wb_absmax,

            "vic_runoff_area_weighted_mm":
                runoff_area_weighted_mm,

            "vic_bottom_moist_area_weighted_mm":
                bottom_area_weighted_mm,
        })


        # ----------------------------------------------------
        # Strict daily checks
        # ----------------------------------------------------

        if not np.isclose(
            api_flow,
            Q_m3_day,
            rtol=1.0e-12,
            atol=1.0e-7,
        ):

            raise RuntimeError(
                f"API flow mismatch on day {n + 1}"
            )


        if not np.isclose(
            mf6_storage_delta,
            expected_volume,
            rtol=1.0e-9,
            atol=1.0e-4,
        ):

            raise RuntimeError(
                f"Storage-volume mismatch on day {n + 1}"
            )


        # Next VIC segment starts from today's VIC state.
        init_state = state_file


finally:

    # Store final head before finalizing the shared library.
    final_mf6_head = float(
        head[0]
    )

    mf6.finalize()


# ============================================================
# Save daily coupling trajectory
# ============================================================

summary_file = (
    OUT /
    "coupling_daily.csv"
)


with summary_file.open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Cumulative conservation
# ============================================================

analytical_total_storage = (
    (
        final_mf6_head -
        INITIAL_MF6_HEAD_M
    ) *
    STORAGE_COEFF_M2
)


cumulative_transfer_error = (
    analytical_total_storage -
    cumulative_vic_volume
)


relative_error = (
    cumulative_transfer_error /
    cumulative_vic_volume
    if abs(cumulative_vic_volume) > 0.0
    else 0.0
)


max_daily_transfer_error = max(
    abs(
        row["transfer_error_m3"]
    )
    for row in rows
)

max_api_flow_error = max(
    abs(
        row["api_flow_error_m3_day"]
    )
    for row in rows
)

max_vic_water_error = max(
    row["vic_water_error_absmax_mm"]
    for row in rows
)


print()
print("=" * 76)
print("H3b CLOSED-LOOP SUMMARY")
print("=" * 76)

print(
    "initial MF6 head          =",
    INITIAL_MF6_HEAD_M,
    "m"
)

print(
    "final MF6 head            =",
    final_mf6_head,
    "m"
)

print(
    "total MF6 head change     =",
    final_mf6_head -
    INITIAL_MF6_HEAD_M,
    "m"
)

print()

print(
    "cumulative VIC transfer   =",
    cumulative_vic_volume,
    "m3"
)

print(
    "MF6 storage change        =",
    analytical_total_storage,
    "m3"
)

print(
    "cumulative volume error   =",
    cumulative_transfer_error,
    "m3"
)

print(
    "relative volume error     =",
    relative_error
)

print()

print(
    "max daily transfer error  =",
    max_daily_transfer_error,
    "m3"
)

print(
    "max API flow error        =",
    max_api_flow_error,
    "m3/day"
)

print(
    "max VIC water error       =",
    max_vic_water_error,
    "mm"
)


# ============================================================
# H1/H2b first-step regression check
# ============================================================

expected_h2b_Q = (
    -2466613.9621837637
)

first_Q = rows[0][
    "vic_Q_m3_day"
]

first_step_pass = np.isclose(
    first_Q,
    expected_h2b_Q,
    rtol=0.0,
    atol=1.0e-6,
)


print()
print(
    "DAY-1 H2b REGRESSION:",
    "PASS"
    if first_step_pass
    else "FAIL"
)

print(
    "  H3b day-1 Q =",
    first_Q
)

print(
    "  H2b reference =",
    expected_h2b_Q
)


# ============================================================
# Overall cross-model conservation
# ============================================================

conservation_pass = np.isclose(
    analytical_total_storage,
    cumulative_vic_volume,
    rtol=1.0e-9,
    atol=1.0e-3,
)


print()
print(
    "CUMULATIVE CROSS-MODEL CONSERVATION:",
    "PASS"
    if conservation_pass
    else "FAIL"
)


# ============================================================
# Save compact summary
# ============================================================

overall_summary = {
    "n_coupling_days":
        NDAYS,

    "total_vic_area_m2":
        TOTAL_AREA_M2,

    "mf6_storage_coefficient_m2":
        STORAGE_COEFF_M2,

    "initial_mf6_head_m":
        INITIAL_MF6_HEAD_M,

    "final_mf6_head_m":
        final_mf6_head,

    "total_mf6_head_change_m":
        final_mf6_head -
        INITIAL_MF6_HEAD_M,

    "cumulative_vic_transfer_m3":
        cumulative_vic_volume,

    "mf6_storage_change_m3":
        analytical_total_storage,

    "cumulative_volume_error_m3":
        cumulative_transfer_error,

    "relative_volume_error":
        relative_error,

    "max_daily_transfer_error_m3":
        max_daily_transfer_error,

    "max_api_flow_error_m3_day":
        max_api_flow_error,

    "max_vic_water_error_mm":
        max_vic_water_error,

    "day1_h2b_regression_pass":
        first_step_pass,

    "cumulative_conservation_pass":
        conservation_pass,
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
            overall_summary.keys()
    )

    writer.writeheader()

    writer.writerow(
        overall_summary
    )


# ============================================================
# Documentation
# ============================================================

(OUT / "EXPERIMENT_H3b.md").write_text(
f"""# H3b — First closed two-way VIC–MF6 feedback experiment

Coupling interval
-----------------

1 day for 10 days.

Algorithm
---------

For coupling interval n:

1. Read current MF6 head h[n].
2. Supply h[n] to the VIC fixed-interface hydraulic-head boundary.
3. Advance VIC for one day from restart state S[n].
4. Integrate the signed VIC groundwater exchange over actual VIC cell
   areas.
5. Convert depth exchange to signed volume Q[n].
6. Apply Q[n] to the MF6 GWF-API package with HCOF=0 and RHS=-Q[n].
7. Advance MF6 for one day.
8. Obtain h[n+1].
9. Continue VIC from state S[n+1].

Sign convention
---------------

Positive VIC groundwater exchange:

    VIC -> MF6

Negative VIC groundwater exchange:

    MF6 -> VIC

The same signed Q is the MF6 flow into groundwater.

Geometry
--------

This is an aggregate coupling validation.

All {int(np.sum(valid_area))} active VIC cells are conservatively
aggregated into one MF6 cell having plan area equal to the total VIC
active area:

    {TOTAL_AREA_M2:.17g} m2

The MF6 interface datum is zero. Thus MF6 head is directly supplied as
the VIC groundwater head offset.

This one-cell model is not intended as a final physical groundwater
representation. It isolates and validates closed-loop coupling,
sign conventions, temporal sequencing, and volume conservation.

MF6 storage
-----------

Thickness:

    {THICKNESS_M} m

Specific storage:

    {SS_PER_M} 1/m

Initial head:

    {INITIAL_MF6_HEAD_M} m

VIC interface
-------------

Ka:

    {KA_SCALE} * Ksat3

Effective exchange length:

    {EXCHANGE_LENGTH_M} m

Hydraulic reference:

    soil-column base

Coupling method
---------------

Explicit sequential coupling.

No within-day iteration between VIC and MF6 is performed.
"""
)


print()
print("=" * 76)

if (
    first_step_pass
    and conservation_pass
):

    print(
        "H3b CLOSED TWO-WAY COUPLING: PASS"
    )

else:

    print(
        "H3b CLOSED TWO-WAY COUPLING: "
        "REQUIRES REVIEW"
    )

print("=" * 76)

print()
print("Experiment directory:")
print(OUT)

print()
print("Daily trajectory:")
print(summary_file)

print()
print("Overall summary:")
print(OUT / "summary.csv")
