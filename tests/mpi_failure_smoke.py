#!/usr/bin/env python3

"""intentional worker failure used to verify MPI jobs terminate rather than hang."""

from mpi4py import MPI

world = MPI.COMM_WORLD
rank = world.Get_rank()
if world.Get_size() != 3:
    raise SystemExit("mpi_failure_smoke.py requires exactly 3 ranks")

world.Barrier()
if rank == 1:
    print("[intentional-worker-failure] rank=1 abort_code=17", flush=True)
    world.Abort(17)
world.Barrier()
