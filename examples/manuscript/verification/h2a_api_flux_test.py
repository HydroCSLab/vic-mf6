#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import csv

import numpy as np
from xmipy import XmiWrapper
from mpi4py import MPI


LIBMF6 = (
    str(MF6_LIBRARY)
)

WORK = Path(
    str(VIC_ROOT / 'experiments/gw_exchange/mf6_h2a')
)

OUT = Path(
    str(RESULT_ROOT / 'H2a_api_flux_test')
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Analytical model properties
# ============================================================

INITIAL_HEAD_M = -105.328

AREA_M2 = 1.0
THICKNESS_M = 200.0
SS_PER_M = 0.001

STORAGE = (
    SS_PER_M *
    AREA_M2 *
    THICKNESS_M
)

DT_DAY = 1.0


# ============================================================
# Test both VIC sign directions
#
# q_into_mf6 > 0:
#     recharge into groundwater
#
# q_into_mf6 < 0:
#     groundwater withdrawal / GW -> VIC
# ============================================================

CASES = [
    (
        "vic_to_mf6_recharge",
        +0.2,
    ),
    (
        "mf6_to_vic_discharge",
        -0.2,
    ),
]


def run_case(name, q_into_mf6):

    print()
    print("=" * 68)
    print(name)
    print("=" * 68)

    mf6 = XmiWrapper(
        LIBMF6,
        working_directory=str(WORK),
    )

    mf6.initialize_mpi(MPI.COMM_SELF.py2f())

    try:

        # ----------------------------------------------------
        # Obtain pointers
        # ----------------------------------------------------

        head_addr = mf6.get_var_address(
            "X",
            "H2",
        )

        nbound_addr = mf6.get_var_address(
            "NBOUND",
            "H2",
            "VICAPI",
        )

        nodelist_addr = mf6.get_var_address(
            "NODELIST",
            "H2",
            "VICAPI",
        )

        rhs_addr = mf6.get_var_address(
            "RHS",
            "H2",
            "VICAPI",
        )

        hcof_addr = mf6.get_var_address(
            "HCOF",
            "H2",
            "VICAPI",
        )

        simvals_addr = mf6.get_var_address(
            "SIMVALS",
            "H2",
            "VICAPI",
        )


        head = mf6.get_value_ptr(
            head_addr
        )

        nbound = mf6.get_value_ptr(
            nbound_addr
        )

        nodelist = mf6.get_value_ptr(
            nodelist_addr
        )

        rhs = mf6.get_value_ptr(
            rhs_addr
        )

        hcof = mf6.get_value_ptr(
            hcof_addr
        )

        simvals = mf6.get_value_ptr(
            simvals_addr
        )


        initial_head = float(head[0])

        print(
            "initial head        =",
            initial_head,
            "m",
        )


        # ----------------------------------------------------
        # Analytical expected result
        # ----------------------------------------------------

        expected_dh = (
            q_into_mf6 *
            DT_DAY /
            STORAGE
        )

        expected_head = (
            initial_head +
            expected_dh
        )

        print(
            "prescribed Q into MF6 =",
            q_into_mf6,
            "m3/day",
        )

        print(
            "expected dh           =",
            expected_dh,
            "m",
        )

        print(
            "expected final head   =",
            expected_head,
            "m",
        )


        # ----------------------------------------------------
        # Prepare one MODFLOW time step
        # ----------------------------------------------------

        dt = mf6.get_time_step()

        print(
            "MF6 timestep          =",
            dt,
            "day",
        )


        mf6.prepare_time_step(dt)


        # ----------------------------------------------------
        # Activate one API boundary.
        #
        # NODELIST is MODFLOW's internal Fortran node number.
        # This model contains one cell, therefore node = 1.
        # ----------------------------------------------------

        nbound[0] = 1

        nodelist[0] = 1


        # ----------------------------------------------------
        # Pure specified flux:
        #
        #     Q_into_cell = HCOF*h - RHS
        #
        # Set:
        #
        #     HCOF = 0
        #
        # therefore:
        #
        #     RHS = -Q_into_cell
        # ----------------------------------------------------

        hcof[0] = 0.0

        rhs[0] = -q_into_mf6


        print()
        print("API arrays before solve:")
        print("  NBOUND   =", nbound)
        print("  NODELIST =", nodelist)
        print("  RHS      =", rhs)
        print("  HCOF     =", hcof)


        # ----------------------------------------------------
        # Solve
        # ----------------------------------------------------

        component = 1

        mf6.prepare_solve(component)


        converged = False

        for iteration in range(1, 101):

            # Reapply these values before every nonlinear
            # iteration so the external boundary is explicit.
            nbound[0] = 1
            nodelist[0] = 1
            hcof[0] = 0.0
            rhs[0] = -q_into_mf6

            converged = mf6.solve(
                component
            )

            if converged:
                print(
                    "converged iteration =",
                    iteration,
                )
                break


        if not converged:
            raise RuntimeError(
                "MODFLOW failed to converge."
            )


        mf6.finalize_solve(
            component
        )

        mf6.finalize_time_step()


        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        final_head = float(head[0])

        simulated_api_flow = float(
            simvals[0]
        )

        head_error = (
            final_head -
            expected_head
        )

        flow_error = (
            simulated_api_flow -
            q_into_mf6
        )


        print()
        print("===== RESULTS =====")

        print(
            "final MF6 head        =",
            final_head,
            "m",
        )

        print(
            "expected final head   =",
            expected_head,
            "m",
        )

        print(
            "head error            =",
            head_error,
            "m",
        )

        print(
            "VICAPI SIMVALS        =",
            simulated_api_flow,
            "m3/day",
        )

        print(
            "expected API flow     =",
            q_into_mf6,
            "m3/day",
        )

        print(
            "flow error            =",
            flow_error,
            "m3/day",
        )


        # ----------------------------------------------------
        # Strict validation
        # ----------------------------------------------------

        head_pass = np.isclose(
            final_head,
            expected_head,
            rtol=0.0,
            atol=1.0e-10,
        )

        flow_pass = np.isclose(
            simulated_api_flow,
            q_into_mf6,
            rtol=0.0,
            atol=1.0e-12,
        )


        print()
        print(
            "HEAD TEST:",
            "PASS" if head_pass else "FAIL",
        )

        print(
            "FLOW TEST:",
            "PASS" if flow_pass else "FAIL",
        )


        return {
            "case": name,
            "initial_head_m":
                initial_head,

            "q_into_mf6_m3_day":
                q_into_mf6,

            "rhs_m3_day":
                -q_into_mf6,

            "expected_dh_m":
                expected_dh,

            "expected_final_head_m":
                expected_head,

            "actual_final_head_m":
                final_head,

            "head_error_m":
                head_error,

            "simvals_m3_day":
                simulated_api_flow,

            "flow_error_m3_day":
                flow_error,

            "head_pass":
                head_pass,

            "flow_pass":
                flow_pass,
        }


    finally:

        mf6.finalize()


rows = []

for name, q in CASES:

    rows.append(
        run_case(name, q)
    )


# ============================================================
# Save experiment summary
# ============================================================

summary = OUT / "summary.csv"

with summary.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("=" * 68)
print("H2a SUMMARY")
print("=" * 68)

for row in rows:

    print(
        row["case"],
        "head:",
        "PASS" if row["head_pass"] else "FAIL",
        "flow:",
        "PASS" if row["flow_pass"] else "FAIL",
    )


if not all(
    r["head_pass"] and r["flow_pass"]
    for r in rows
):
    raise RuntimeError(
        "H2a API flux validation failed."
    )


print()
print("H2a ALL TESTS PASSED")
print()
print("Saved:")
print(summary)
