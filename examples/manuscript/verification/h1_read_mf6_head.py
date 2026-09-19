#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT, VIC_ROOT, MF6_LIBRARY
import os

from xmipy import XmiWrapper
from mpi4py import MPI


work = Path(
    str(VIC_ROOT / 'experiments/gw_exchange/mf6_h1')
)


# ------------------------------------------------------------
# Locate libmf6.so
# ------------------------------------------------------------

candidates = [MF6_LIBRARY]


print("libmf6 candidates:")

for candidate in candidates:
    print("  ", candidate)


libmf6 = candidates[0]

print()
print("Using:")
print(libmf6)


# ------------------------------------------------------------
# Initialize MODFLOW through XMI
# ------------------------------------------------------------

mf6 = XmiWrapper(
    str(libmf6),
    working_directory=str(work)
)


mf6.initialize_mpi(MPI.COMM_SELF.py2f())


try:

    address = None

    for component in (
        "H1",
        "h1",
    ):

        try:

            address = mf6.get_var_address(
                "X",
                component
            )

            print()
            print(
                "HEAD memory address:",
                address
            )

            break

        except Exception:
            pass


    if address is None:
        raise RuntimeError(
            "Could not locate H1/X head variable."
        )


    head = mf6.get_value_ptr(address)

    print()
    print("===== MF6 HEAD THROUGH XMI =====")
    print("shape =", head.shape)
    print("head  =", head)

    head_value = float(head[0])

    outfile = work / "mf6_head_offset.txt"

    outfile.write_text(
        f"{head_value:.15g}\n"
    )

    print()
    print("Head offset written to:")
    print(outfile)


finally:

    mf6.finalize()
