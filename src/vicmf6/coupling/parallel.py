"""MPI collectives used by the explicit scheme.

Every outer rank must enter these operations in the same order, including the
controller, which contributes neutral values. SUM combines extensive amounts;
MAX reports worst errors or slowest workers. Packing related scalars reduces
latency without changing their operations or introducing asynchronous state.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..exchange import SignedVolume
from ..mf6 import Mf6AdvanceResult
from .records import GroundwaterStatistics, WindowTimings


class CouplingCommunicator:
    """Communicate through the outer world; MF6 owns a separate worker group."""

    def __init__(self, world: Any, mpi: Any) -> None:
        self.world = world
        self.mpi = mpi
        self.rank = int(world.Get_rank())
        self.is_controller = self.rank == 0
        self.elapsed_seconds = 0.0

    def reduce_array(self, values: np.ndarray, operation: Any) -> np.ndarray | None:
        receive = np.empty_like(values) if self.is_controller else None
        start = time.perf_counter()
        self.world.Reduce(values, receive, op=operation, root=0)
        self.elapsed_seconds += time.perf_counter() - start
        return receive

    def broadcast_vic_depth(self, depth_mm: np.ndarray) -> None:
        start = time.perf_counter()
        self.world.Bcast(depth_mm, root=0)
        self.elapsed_seconds += time.perf_counter() - start

    def wait_for_conservation_check(self) -> None:
        """Do not start the next native solve until the controller accepts mapping."""
        start = time.perf_counter()
        self.world.Barrier()
        self.elapsed_seconds += time.perf_counter() - start

    def reduce_signed_volumes(
        self, *volumes: SignedVolume
    ) -> tuple[SignedVolume, ...] | None:
        send = np.asarray(
            [(v.positive_m3, v.negative_m3, v.net_m3) for v in volumes],
            dtype=np.float64,
        )
        receive = self.reduce_array(send, self.mpi.SUM)
        if receive is None:
            return None
        return tuple(SignedVolume(*map(float, row)) for row in receive)

    def reduce_groundwater_statistics(
        self, advance: Mf6AdvanceResult | None, timings: WindowTimings
    ) -> GroundwaterStatistics | None:
        # Use the head snapshot returned by the completed solve. Reading native
        # heads again would copy the same node-sized array a second time.
        head = advance.head_m if advance is not None else np.empty(0)
        lateral = advance.lateral if advance is not None else None
        totals = self.reduce_array(
            np.asarray(
                [
                    int(lateral is not None),
                    lateral.domain_net_m3 if lateral else 0.0,
                    lateral.gross_pair_volume_m3 if lateral else 0.0,
                    head.sum(dtype=np.float64),
                    head.size,
                ],
                dtype=np.float64,
            ),
            self.mpi.SUM,
        )
        maxima = self.reduce_array(
            np.asarray(
                [
                    advance.maximum_api_error_m3_per_day if advance else 0.0,
                    advance.nonlinear_iterations if advance else 0,
                    lateral.maximum_pair_antisymmetry_m3 if lateral else 0.0,
                    head.max(initial=-np.inf),
                    timings.mapping_seconds,
                    timings.mf6_seconds,
                    self.elapsed_seconds,
                ],
                dtype=np.float64,
            ),
            self.mpi.MAX,
        )
        minimum = self.reduce_array(
            np.asarray([head.min(initial=np.inf)]), self.mpi.MIN
        )
        if not self.is_controller:
            return None
        assert totals is not None and maxima is not None and minimum is not None
        lateral_count, lateral_net, lateral_gross, head_sum, head_count = totals
        (
            api_error,
            iterations,
            pair_error,
            head_max,
            mapping_time,
            mf6_time,
            mpi_time,
        ) = maxima
        assert head_count > 0
        return GroundwaterStatistics(
            maximum_api_error=float(api_error),
            maximum_iterations=int(iterations),
            lateral_count=int(lateral_count),
            lateral_domain_net=float(lateral_net),
            lateral_gross=float(lateral_gross),
            lateral_pair_error=float(pair_error),
            head_min=float(minimum[0]),
            head_max=float(head_max),
            head_mean=float(head_sum / head_count),
            mapping_seconds=float(mapping_time),
            mf6_seconds=float(mf6_time),
            mpi_seconds=float(mpi_time),
        )
