"""Own model lifetimes and static geometry for one MPI run.

Rank zero owns VIC restarts, full geometry, and files. Each worker owns one
persistent MF6 model and only that model's overlaps. All ranks retain the same
VIC cell ordering so time-varying broadcasts never need to send identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ApplicationConfig
from ..diagnostics import DiagnosticsWriter
from ..errors import CouplingRuntimeError
from ..exchange import ExchangeTable
from ..mf6 import Mf6Runtime
from ..vic import VicRuntime
from .parallel import CouplingCommunicator


@dataclass
class CouplingSession:
    config: ApplicationConfig
    logger: object
    parallel: CouplingCommunicator
    worker_comm: Any
    model_names: list[str]
    exchange_table: ExchangeTable
    head_overlap_area_m2: np.ndarray | None
    mf6: Mf6Runtime | None = None
    vic: VicRuntime | None = None
    diagnostics: DiagnosticsWriter | None = None
    previous_vic_state: Path | None = None

    @classmethod
    def initialize(
        cls, config: ApplicationConfig, logger: object, mpi: Any
    ) -> CouplingSession:
        parallel = CouplingCommunicator(mpi.COMM_WORLD, mpi)
        world = parallel.world
        model_names = world.bcast(
            [model.name for model in config.mf6_source.models]
            if parallel.is_controller
            else None,
            root=0,
        )
        expected_size = len(model_names) + 1
        if world.Get_size() != expected_size:
            raise CouplingRuntimeError(
                "outer MPI size must equal one controller plus one rank per GWF model: "
                f"expected={expected_size} actual={world.Get_size()} models={model_names}"
            )

        table = None
        partitions = None
        if parallel.is_controller:
            table = ExchangeTable.from_csv(
                config.coupling.exchange_table,
                require_full_vic_coverage=config.coupling.require_full_vic_coverage,
                coverage_relative_tolerance=config.coupling.coverage_relative_tolerance,
            )
            if {name.casefold() for name in model_names} != {
                name.casefold() for name in table.model_names
            }:
                raise CouplingRuntimeError(
                    "exchange-table model names do not match GWF models in mfsim.nam: "
                    f"table={table.model_names} mf6={model_names}"
                )
            # Parse and validate the CSV once. Each worker receives local overlap
            # rows instead of reading and retaining every model's geometry.
            partitions = [None] + [table.for_model(name) for name in model_names]
        local_table = world.scatter(partitions, root=0)
        if not parallel.is_controller:
            table = local_table
        assert table is not None
        worker_comm = world.Split(
            1 if not parallel.is_controller else mpi.UNDEFINED, parallel.rank
        )
        session = cls(config, logger, parallel, worker_comm, model_names, table, None)

        if parallel.is_controller:
            session.vic = VicRuntime(config.vic, table, logger=logger)
            session.previous_vic_state = session.vic.initial_state()
            session.diagnostics = DiagnosticsWriter(
                config.coupling.diagnostics_directory
            )
            session.diagnostics.write_manifest(
                config,
                world_size=world.Get_size(),
                mf6_models=model_names,
            )
            local_area = np.zeros(table.vic_cell_count, dtype=np.float64)
        else:
            model_name = model_names[parallel.rank - 1]
            session.mf6 = Mf6Runtime(
                config.mf6,
                model_name=model_name,
                coupled_nodes=table.coupled_nodes(model_name),
                logger=logger,
            )
            session.mf6.initialize(worker_comm.py2f())
            local_area = table.head_overlap_area_for_model(model_name)

        # Grid geometry stays fixed throughout the run. The area denominator is
        # reduced once; only head * area needs a new reduction in each window.
        session.head_overlap_area_m2 = parallel.reduce_array(local_area, mpi.SUM)
        return session

    def finalize_successful_run(self) -> None:
        """Release native resources only after all windows complete successfully.

        Native finalization and communicator release can be collective. If one
        rank fails, entering them in a finally block can deadlock before the CLI
        reaches MPI.Abort. Failures must propagate directly to that abort path.
        """
        if self.mf6 is not None:
            self.mf6.finalize()
            self.worker_comm.Free()
