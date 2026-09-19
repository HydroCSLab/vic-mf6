#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv
import math

import netCDF4 as nc
import numpy as np

from xmipy import XmiWrapper
from mpi4py import MPI


# ============================================================
# Paths
# ============================================================

LIBMF6 = Path(
    str(MF6_LIBRARY)
)

VIC_FILE = Path(
    str(RESULT_ROOT / 'H1_mf6_to_vic_head/fluxes.mf6_head.nc')
)

DOMAIN_FILE = Path(
    str(SAMPLE_ROOT / 'parameters/domain.stehekin.20151028.nc')
)

WORK = Path(
    str(VIC_ROOT / 'experiments/gw_exchange/mf6_h2b')
)

OUT = Path(
    str(RESULT_ROOT / 'H2b_vic_flux_to_mf6')
)

WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# MF6 controlled storage properties
# ============================================================

INITIAL_HEAD_M = -105.328

TOP_M = 0.0
BOT_M = -200.0

THICKNESS_M = TOP_M - BOT_M

SS_PER_M = 0.001

DT_DAY = 1.0


# ============================================================
# Read actual VIC day-1 groundwater exchange
# ============================================================

with nc.Dataset(VIC_FILE) as vds:

    q = np.ma.asarray(
        vds["OUT_GW_EXCHANGE"][0, :, :]
    )

    lat = np.asarray(vds["lat"][:])
    lon = np.asarray(vds["lon"][:])


with nc.Dataset(DOMAIN_FILE) as dds:

    area_var = dds["area"]

    area = np.ma.asarray(
        area_var[:, :]
    )

    domain_mask = (
        np.asarray(dds["mask"][:]) > 0
    )

    area_units = getattr(
        area_var,
        "units",
        ""
    )


print()
print("===== VIC AREA INFORMATION =====")
print("domain area units =", repr(area_units))


# ============================================================
# Normalize area to m2
# ============================================================

u = (
    str(area_units)
    .strip()
    .lower()
    .replace(" ", "")
)


if u in (
    "m2",
    "m^2",
    "m**2",
    "squaremeters",
    "squaremetres",
):

    area_m2 = np.ma.asarray(area)


elif u in (
    "km2",
    "km^2",
    "km**2",
    "squarekilometers",
    "squarekilometres",
):

    area_m2 = np.ma.asarray(area) * 1.0e6


else:

    raise RuntimeError(
        "Unsupported or missing DOMAIN area units: "
        f"{area_units!r}. "
        "Inspect the domain NetCDF before continuing."
    )


# ============================================================
# Valid VIC cells
# ============================================================

valid = (
    domain_mask
    & ~np.ma.getmaskarray(q)
    & ~np.ma.getmaskarray(area_m2)
)


q_mm = np.asarray(q)[valid]

a_m2 = np.asarray(area_m2)[valid]


if q_mm.size == 0:
    raise RuntimeError(
        "No active VIC cells found."
    )


# ============================================================
# Convert VIC depth exchange to MF6 volumetric rate
#
# VIC:
#
#     q > 0 : VIC -> groundwater
#     q < 0 : groundwater -> VIC
#
# Therefore the same sign can be used for Q_into_MF6.
#
# Day-1 VIC output is a daily summed amount [mm].
# Since the MF6 test step is exactly one day, its numerical
# value is also the equivalent average rate [mm/day].
# ============================================================

cell_Q_m3_day = (
    q_mm *
    1.0e-3 *
    a_m2
)

Q_total_m3_day = float(
    np.sum(cell_Q_m3_day)
)

A_total_m2 = float(
    np.sum(a_m2)
)

q_area_weighted_mm_day = (
    Q_total_m3_day /
    A_total_m2 *
    1000.0
)


print()
print("===== VIC -> MF6 VOLUME MAPPING =====")
print("active VIC cells          =", q_mm.size)
print("total VIC area            =", A_total_m2, "m2")
print(
    "area-weighted VIC q       =",
    q_area_weighted_mm_day,
    "mm/day"
)
print(
    "total VIC exchange volume =",
    Q_total_m3_day,
    "m3/day"
)


# ============================================================
# Save per-cell mapping
# ============================================================

rows = []

jj, ii = np.where(valid)

for n, (j, i) in enumerate(zip(jj, ii)):

    rows.append({
        "lat": float(lat[j]),
        "lon": float(lon[i]),
        "area_m2": float(area_m2[j, i]),
        "vic_q_mm_day": float(q[j, i]),
        "mf6_Q_m3_day": float(
            q[j, i] *
            1.0e-3 *
            area_m2[j, i]
        ),
    })


with (OUT / "vic_cell_fluxes.csv").open(
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
# Construct one MF6 cell with exactly the total VIC plan area
# ============================================================

CELL_WIDTH_M = math.sqrt(
    A_total_m2
)


(WORK / "mfsim.nam").write_text(
"""BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6 h2b.tdis
END TIMING

BEGIN MODELS
  GWF6 h2b.nam H2B
END MODELS

BEGIN EXCHANGES
END EXCHANGES

BEGIN SOLUTIONGROUP 1
  IMS6 h2b.ims H2B
END SOLUTIONGROUP
"""
)


(WORK / "h2b.tdis").write_text(
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


(WORK / "h2b.ims").write_text(
"""BEGIN OPTIONS
  PRINT_OPTION SUMMARY
  COMPLEXITY SIMPLE
END OPTIONS
"""
)


(WORK / "h2b.nam").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN PACKAGES
  DIS6 h2b.dis DIS
  IC6  h2b.ic  IC
  NPF6 h2b.npf NPF
  STO6 h2b.sto STO
  API6 h2b.api VICAPI
  OC6  h2b.oc  OC
END PACKAGES
"""
)


(WORK / "h2b.dis").write_text(
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


(WORK / "h2b.ic").write_text(
f"""BEGIN GRIDDATA
  STRT
    CONSTANT {INITIAL_HEAD_M:.15g}
END GRIDDATA
"""
)


(WORK / "h2b.npf").write_text(
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


(WORK / "h2b.sto").write_text(
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


(WORK / "h2b.api").write_text(
"""BEGIN OPTIONS
  SAVE_FLOWS
END OPTIONS

BEGIN DIMENSIONS
  MAXBOUND 1
END DIMENSIONS
"""
)


(WORK / "h2b.oc").write_text(
"""BEGIN OPTIONS
  HEAD FILEOUT h2b.hds
  BUDGET FILEOUT h2b.cbc
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
# Analytical expected storage response
# ============================================================

storage_coefficient_m2 = (
    SS_PER_M *
    A_total_m2 *
    THICKNESS_M
)

expected_volume_m3 = (
    Q_total_m3_day *
    DT_DAY
)

expected_dh_m = (
    expected_volume_m3 /
    storage_coefficient_m2
)

expected_head_m = (
    INITIAL_HEAD_M +
    expected_dh_m
)


print()
print("===== ANALYTICAL MF6 EXPECTATION =====")
print(
    "MF6 plan area             =",
    A_total_m2,
    "m2"
)
print(
    "cell width                =",
    CELL_WIDTH_M,
    "m"
)
print(
    "storage coefficient       =",
    storage_coefficient_m2,
    "m2"
)
print(
    "expected exchange volume  =",
    expected_volume_m3,
    "m3"
)
print(
    "expected head change      =",
    expected_dh_m,
    "m"
)
print(
    "expected final head       =",
    expected_head_m,
    "m"
)


# ============================================================
# Run MF6 through XMI
# ============================================================

mf6 = XmiWrapper(
    str(LIBMF6),
    working_directory=str(WORK)
)

mf6.initialize_mpi(MPI.COMM_SELF.py2f())


try:

    head = mf6.get_value_ptr(
        mf6.get_var_address(
            "X",
            "H2B"
        )
    )

    nbound = mf6.get_value_ptr(
        mf6.get_var_address(
            "NBOUND",
            "H2B",
            "VICAPI"
        )
    )

    nodelist = mf6.get_value_ptr(
        mf6.get_var_address(
            "NODELIST",
            "H2B",
            "VICAPI"
        )
    )

    rhs = mf6.get_value_ptr(
        mf6.get_var_address(
            "RHS",
            "H2B",
            "VICAPI"
        )
    )

    hcof = mf6.get_value_ptr(
        mf6.get_var_address(
            "HCOF",
            "H2B",
            "VICAPI"
        )
    )

    simvals = mf6.get_value_ptr(
        mf6.get_var_address(
            "SIMVALS",
            "H2B",
            "VICAPI"
        )
    )


    actual_initial_head = float(
        head[0]
    )


    print()
    print("===== XMI INITIAL STATE =====")
    print(
        "MF6 initial head =",
        actual_initial_head,
        "m"
    )


    # --------------------------------------------------------
    # Explicit one-day step.
    #
    # We deliberately pass 1.0 rather than using get_time_step()
    # before prepare_time_step(), because your current MF6/XMI
    # build reports 0.0 at that pre-step point.
    # --------------------------------------------------------

    mf6.prepare_time_step(
        DT_DAY
    )


    nbound[0] = 1
    nodelist[0] = 1

    hcof[0] = 0.0

    # Q_into_MF6 = -RHS
    rhs[0] = -Q_total_m3_day


    print()
    print("===== API BOUNDARY =====")
    print("NBOUND   =", nbound)
    print("NODELIST =", nodelist)
    print("HCOF     =", hcof)
    print("RHS      =", rhs)
    print(
        "Q into MF6 =",
        Q_total_m3_day,
        "m3/day"
    )


    component = 1

    mf6.prepare_solve(
        component
    )


    converged = False

    for iteration in range(
        1,
        101
    ):

        # Reapply explicit external boundary every solve iteration.
        nbound[0] = 1
        nodelist[0] = 1
        hcof[0] = 0.0
        rhs[0] = -Q_total_m3_day

        converged = mf6.solve(
            component
        )

        if converged:

            print(
                "converged iteration =",
                iteration
            )

            break


    if not converged:

        raise RuntimeError(
            "MF6 failed to converge."
        )


    mf6.finalize_solve(
        component
    )

    mf6.finalize_time_step()


    # ========================================================
    # Validation
    # ========================================================

    final_head_m = float(
        head[0]
    )

    api_flow_m3_day = float(
        simvals[0]
    )

    actual_dh_m = (
        final_head_m -
        actual_initial_head
    )

    volume_from_head_m3 = (
        actual_dh_m *
        storage_coefficient_m2
    )

    head_error_m = (
        final_head_m -
        expected_head_m
    )

    api_flow_error_m3_day = (
        api_flow_m3_day -
        Q_total_m3_day
    )

    volume_error_m3 = (
        volume_from_head_m3 -
        expected_volume_m3
    )

    relative_volume_error = (
        volume_error_m3 /
        expected_volume_m3
        if abs(expected_volume_m3) > 0.0
        else 0.0
    )


    print()
    print("===== H2b RESULTS =====")

    print(
        "actual final head       =",
        final_head_m,
        "m"
    )

    print(
        "expected final head     =",
        expected_head_m,
        "m"
    )

    print(
        "head error              =",
        head_error_m,
        "m"
    )

    print()

    print(
        "VIC integrated volume   =",
        expected_volume_m3,
        "m3"
    )

    print(
        "MF6 storage volume      =",
        volume_from_head_m3,
        "m3"
    )

    print(
        "volume error            =",
        volume_error_m3,
        "m3"
    )

    print(
        "relative volume error   =",
        relative_volume_error
    )

    print()

    print(
        "VICAPI SIMVALS          =",
        api_flow_m3_day,
        "m3/day"
    )

    print(
        "expected API flow       =",
        Q_total_m3_day,
        "m3/day"
    )

    print(
        "API flow error          =",
        api_flow_error_m3_day,
        "m3/day"
    )


    # ========================================================
    # Pass criteria
    # ========================================================

    head_pass = np.isclose(
        final_head_m,
        expected_head_m,
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    flow_pass = np.isclose(
        api_flow_m3_day,
        Q_total_m3_day,
        rtol=1.0e-12,
        atol=1.0e-8,
    )

    volume_pass = np.isclose(
        volume_from_head_m3,
        expected_volume_m3,
        rtol=1.0e-9,
        atol=1.0e-6,
    )


    print()
    print(
        "HEAD TEST:",
        "PASS" if head_pass else "FAIL"
    )

    print(
        "API FLOW TEST:",
        "PASS" if flow_pass else "FAIL"
    )

    print(
        "VOLUME CONSERVATION TEST:",
        "PASS" if volume_pass else "FAIL"
    )


    summary = {
        "n_vic_cells":
            int(q_mm.size),

        "total_vic_area_m2":
            A_total_m2,

        "area_weighted_vic_q_mm_day":
            q_area_weighted_mm_day,

        "vic_total_Q_m3_day":
            Q_total_m3_day,

        "initial_mf6_head_m":
            actual_initial_head,

        "expected_head_change_m":
            expected_dh_m,

        "expected_final_head_m":
            expected_head_m,

        "actual_final_head_m":
            final_head_m,

        "head_error_m":
            head_error_m,

        "mf6_simvals_m3_day":
            api_flow_m3_day,

        "api_flow_error_m3_day":
            api_flow_error_m3_day,

        "vic_volume_m3":
            expected_volume_m3,

        "mf6_storage_volume_m3":
            volume_from_head_m3,

        "volume_error_m3":
            volume_error_m3,

        "relative_volume_error":
            relative_volume_error,

        "head_pass":
            bool(head_pass),

        "flow_pass":
            bool(flow_pass),

        "volume_pass":
            bool(volume_pass),
    }


    with (OUT / "summary.csv").open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary.keys()
        )

        writer.writeheader()
        writer.writerow(summary)


    with (OUT / "EXPERIMENT_H2b.md").open(
        "w"
    ) as f:

        f.write(
            "# H2b — VIC groundwater exchange to MF6\n\n"
        )

        f.write(
            "Day-1 groundwater exchange from the H1 VIC run "
            "was converted cell-by-cell from mm/day to m3/day "
            "using actual VIC domain areas.\n\n"
        )

        f.write(
            "All active VIC-cell volumes were conservatively "
            "aggregated into one MF6 API boundary.\n\n"
        )

        f.write(
            "Sign convention:\n\n"
        )

        f.write(
            "- positive VIC exchange = recharge into MF6\n"
            "- negative VIC exchange = groundwater withdrawal\n\n"
        )

        f.write(
            "MF6 API mapping:\n\n"
        )

        f.write(
            "`HCOF = 0`\n\n"
        )

        f.write(
            "`RHS = -Q_VIC`\n\n"
        )

        f.write(
            "The MF6 cell plan area equals the summed active "
            "VIC-cell area, permitting an independent storage-"
            "volume conservation check.\n"
        )


    if not (
        head_pass
        and flow_pass
        and volume_pass
    ):

        raise RuntimeError(
            "H2b validation failed."
        )


finally:

    mf6.finalize()


print()
print("========================================")
print("H2b ALL TESTS PASSED")
print("========================================")

print()
print("Saved:")
print(OUT / "vic_cell_fluxes.csv")
print(OUT / "summary.csv")
print(OUT / "EXPERIMENT_H2b.md")
