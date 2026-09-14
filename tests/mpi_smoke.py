#!/usr/bin/env python3

"""small deterministic controller/worker collective test without model binaries."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from vicmf6.exchange import ExchangeTable

world = MPI.COMM_WORLD
rank = world.Get_rank()
if world.Get_size() != 3:
    raise SystemExit("mpi_smoke.py requires exactly 3 ranks")

records = [
    {
        "vic_id": "A",
        "vic_row": 0,
        "vic_col": 0,
        "vic_area_m2": 2.0,
        "mf6_model": "LEFT",
        "mf6_node": 1,
        "overlap_area_m2": 1.0,
    },
    {
        "vic_id": "A",
        "vic_row": 0,
        "vic_col": 0,
        "vic_area_m2": 2.0,
        "mf6_model": "RIGHT",
        "mf6_node": 1,
        "overlap_area_m2": 1.0,
    },
]
table = ExchangeTable.from_records(records)
local_num = np.zeros(1, dtype=float)
local_area = np.zeros(1, dtype=float)
if rank == 1:
    contribution = table.head_contribution_for_model("LEFT", np.array([-100.0]))
    local_num[:] = contribution.head_area_sum_m3
    local_area[:] = contribution.area_sum_m2
elif rank == 2:
    contribution = table.head_contribution_for_model("RIGHT", np.array([-120.0]))
    local_num[:] = contribution.head_area_sum_m3
    local_area[:] = contribution.area_sum_m2

num = np.empty(1, dtype=float) if rank == 0 else None
area = np.empty(1, dtype=float) if rank == 0 else None
world.Reduce(local_num, num, op=MPI.SUM, root=0)
world.Reduce(local_area, area, op=MPI.SUM, root=0)

if rank == 0:
    mapped = table.finish_head_mapping(num, area)
    np.testing.assert_array_equal(mapped, np.array([-110.0]))
    print("[OK] MPI controller/worker reverse-map smoke test")
