"""Exercise the real explicit driver and MPI with deterministic in-memory models.

Run through test_mpi_coupling.py. Three outer ranks include two groundwater
partitions, unequal overlap weights, opposing fluxes, and two windows. Native
physics is verified separately by the complete Stehekin acceptance workflow.
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from vicmf6.coupling import session
from vicmf6.driver import run_coupling
from vicmf6.exchange import SignedVolume
from vicmf6.mf6 import LateralFlowDiagnostics, Mf6AdvanceResult

world = MPI.COMM_WORLD
rank = world.Get_rank()
scenario, output_path = sys.argv[1], Path(sys.argv[2])
initial_heads = np.array([-10.0, -20.0, -30.0, -40.0])
node_volumes = np.array([0.014, -0.024, 0.040, 0.025])


class GroundwaterModel:
    def __init__(self, config, *, model_name, coupled_nodes, logger):
        self.model_name = model_name
        self.head = initial_heads[2 * (rank - 1) : 2 * rank].copy()
        self.node_count = 2

    def initialize(self, comm_handle):
        assert MPI.Comm.f2py(comm_handle).Get_size() == 2

    def current_head(self):
        return self.head.copy()

    def advance_to(self, target, volumes, **kwargs):
        if scenario == "worker-failure" and rank == 1:
            raise RuntimeError("injected worker failure")
        np.testing.assert_allclose(
            volumes, node_volumes[2 * (rank - 1) : 2 * rank], rtol=0, atol=1e-16
        )
        self.head += volumes
        lateral = LateralFlowDiagnostics(np.zeros(2), 0.0, float(rank), rank * 1e-13)
        if scenario == "partial-topology" and rank == 2:
            lateral = None
        return Mf6AdvanceResult(
            self.head.copy(),
            SignedVolume.from_values(volumes),
            SignedVolume.from_values(volumes),
            rank * 1e-12,
            rank + 1,
            lateral,
        )

    def finalize(self):
        # The failing worker must reach Abort without entering this path.
        (output_path / f"finalized-{rank}").touch()


class SurfaceModel:
    def __init__(self, config, table, *, logger):
        assert table.overlap_count == 5

    def initial_state(self):
        return None

    def prepare_window(self, window, head, *, previous_state):
        expected_heads = initial_heads + window.index * node_volumes
        expected = [
            (2 * expected_heads[0] + 8 * expected_heads[2]) / 10,
            (6 * expected_heads[1] + 4 * expected_heads[2]) / 10,
            expected_heads[3],
        ]
        np.testing.assert_allclose(head, expected, rtol=0, atol=1e-12)
        assert previous_state == (None if window.index == 0 else Path("restart-0"))
        return window

    def run_window(self, window):
        return SimpleNamespace(
            state_file=Path(f"restart-{window.index}"),
            spawn_seconds=0.0,
            output_read_seconds=0.0,
            maximum_water_error_mm=0.0,
            exchange_grid_mm=np.array([[7.0, -4.0, 2.5]]),
        )


class Diagnostics:
    def __init__(self, path):
        self.rows = []

    def write_manifest(self, config, *, world_size, mf6_models):
        assert world_size == 3 and mf6_models == ["A", "B"]

    def append(self, row):
        for actual, expected in [
            (row.vic_positive_m3, 0.095),
            (row.vic_negative_m3, -0.04),
            (row.boundary_positive_m3, 0.079),
            (row.boundary_negative_m3, -0.024),
            (row.applied_net_m3, 0.055),
        ]:
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-16)
        heads = initial_heads + row.step * node_volumes
        np.testing.assert_allclose(
            [row.head_min_m, row.head_mean_m, row.head_max_m],
            [heads.min(), heads.mean(), heads.max()],
            rtol=0,
            atol=1e-12,
        )
        assert row.nonlinear_iterations_max == 3
        assert row.maximum_api_rate_error_m3_per_day == 2e-12
        if scenario == "partial-topology":
            assert row.lateral_domain_net_m3 is None
            assert row.lateral_gross_pair_m3 is None
        else:
            assert row.lateral_domain_net_m3 == 0.0
            assert row.lateral_gross_pair_m3 == 3.0
            assert row.lateral_maximum_pair_antisymmetry_m3 == 2e-13
        self.rows.append(row)

    def write_summary(self, **kwargs):
        assert len(self.rows) == 2


def main():
    if world.Get_size() != 3:
        raise ValueError("this scenario requires three outer MPI ranks")
    table_path = output_path / "overlaps.csv"
    if rank == 0:
        records = [
            dict(
                vic_id=f"V{vic}",
                vic_row=0,
                vic_col=vic,
                vic_area_m2=10,
                mf6_model=model,
                mf6_node=node,
                overlap_area_m2=area,
            )
            for vic, model, node, area in (
                (0, "A", 1, 2),
                (0, "B", 1, 8),
                (1, "B", 1, 4),
                (1, "A", 2, 6),
                (2, "B", 2, 10),
            )
        ]
        with table_path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    world.Barrier()
    start = datetime(2000, 1, 1)
    config = SimpleNamespace(
        mf6_source=SimpleNamespace(
            models=[SimpleNamespace(name=name) for name in ("A", "B")]
        ),
        mf6=SimpleNamespace(start_time=start),
        vic=SimpleNamespace(),
        coupling=SimpleNamespace(
            exchange_table=table_path,
            require_full_vic_coverage=True,
            coverage_relative_tolerance=1e-12,
            diagnostics_directory=output_path,
            start_time=start,
            end_time=start + timedelta(days=2),
            interval_days=1,
            conservation_absolute_tolerance_m3=1e-12,
            conservation_relative_tolerance=1e-12,
            api_absolute_tolerance_m3_per_day=1e-6,
        ),
    )
    session.Mf6Runtime = GroundwaterModel
    session.VicRuntime = SurfaceModel
    session.DiagnosticsWriter = Diagnostics
    run_coupling(config, logger=object())
    world.Barrier()
    if rank == 0:
        assert all((output_path / f"finalized-{worker}").is_file() for worker in (1, 2))
        print(f"PASS: {scenario}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"rank {rank}: {exc}", file=sys.stderr, flush=True)
        world.Abort(1)
