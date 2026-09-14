"""high-level explicit coupling loop; comments narrate the scientific algorithm."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import ApplicationConfig, create_run_directories
from .diagnostics import DiagnosticsWriter, make_window_diagnostics
from .errors import CouplingRuntimeError
from .exchange import (
    ExchangeTable,
    SignedVolume,
    assert_net_volume_close,
    assert_signed_volume_close,
)
from .mf6 import Mf6Runtime
from .schedule import build_windows
from .vic import VicRuntime


def run_coupling(config: ApplicationConfig, *, logger: object) -> int:
    """run the explicit two-way VIC-MF6 scheme under the current MPI world."""

    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise CouplingRuntimeError(
            "mpi4py is required for coupled execution; install vicmf6[runtime]"
        ) from exc

    world = MPI.COMM_WORLD
    rank = int(world.Get_rank())
    world_size = int(world.Get_size())
    worker_comm = world.Split(1 if rank > 0 else MPI.UNDEFINED, rank)

    exchange_table = ExchangeTable.from_csv(
        config.coupling.exchange_table,
        require_full_vic_coverage=config.coupling.require_full_vic_coverage,
        coverage_relative_tolerance=config.coupling.coverage_relative_tolerance,
    )

    model_names = None
    if rank == 0:
        create_run_directories(config)
        model_names = [model.name for model in config.mf6_source.models]
    model_names = world.bcast(model_names, root=0)

    expected_world_size = len(model_names) + 1
    if world_size != expected_world_size:
        raise CouplingRuntimeError(
            "outer MPI size must equal one controller plus one rank per GWF model: "
            f"expected={expected_world_size} actual={world_size} models={model_names}"
        )
    if {name.casefold() for name in model_names} != {
        name.casefold() for name in exchange_table.model_names
    }:
        raise CouplingRuntimeError(
            "exchange-table model names do not match GWF models in mfsim.nam: "
            f"table={exchange_table.model_names} mf6={model_names}"
        )

    mf6_runtime: Mf6Runtime | None = None
    vic_runtime: VicRuntime | None = None
    diagnostics: DiagnosticsWriter | None = None
    previous_vic_state = None

    # groundwater workers are initialized once before the coupling loop. this
    # is the key lifecycle property preserved from the proven one-way coupler:
    # mf6 head and storage persist while VIC is restarted in controlled windows.
    if rank > 0:
        model_name = model_names[rank - 1]
        mf6_runtime = Mf6Runtime(
            config.mf6,
            model_name=model_name,
            coupled_nodes=exchange_table.coupled_nodes(model_name),
            logger=logger,
        )
        mf6_runtime.initialize(worker_comm.py2f())
    else:
        vic_runtime = VicRuntime(config.vic, exchange_table, logger=logger)
        previous_vic_state = vic_runtime.initial_state()
        diagnostics = DiagnosticsWriter(config.coupling.diagnostics_directory)
        diagnostics.write_manifest(
            config,
            world_size=world_size,
            mf6_models=model_names,
        )

    windows = build_windows(
        config.coupling.start_time,
        config.coupling.end_time,
        config.coupling.interval_days,
    )
    run_start = time.perf_counter()

    try:
        for window in windows:
            window_start_wall = time.perf_counter()
            local_mapping_seconds = 0.0
            local_mpi_seconds = 0.0
            local_mf6_seconds = 0.0
            vic_prepare_seconds = 0.0
            vic_seconds = 0.0
            io_seconds = 0.0

            # map the current distributed groundwater head back to VIC. each
            # worker contributes head*overlap_area and overlap_area separately;
            # reducing those two extensive quantities makes the result invariant
            # to how MF6 models are partitioned across workers.
            map_start = time.perf_counter()
            if rank > 0:
                assert mf6_runtime is not None
                head_contribution = exchange_table.head_contribution_for_model(
                    model_names[rank - 1], mf6_runtime.current_head()
                )
                local_head_numerator = head_contribution.head_area_sum_m3
                local_head_area = head_contribution.area_sum_m2
            else:
                local_head_numerator = np.zeros(
                    exchange_table.vic_cell_count, dtype=np.float64
                )
                local_head_area = np.zeros(
                    exchange_table.vic_cell_count, dtype=np.float64
                )
            local_mapping_seconds += time.perf_counter() - map_start

            reduced_head_numerator = (
                np.empty(exchange_table.vic_cell_count, dtype=np.float64)
                if rank == 0
                else None
            )
            reduced_head_area = (
                np.empty(exchange_table.vic_cell_count, dtype=np.float64)
                if rank == 0
                else None
            )
            mpi_start = time.perf_counter()
            world.Reduce(
                local_head_numerator,
                reduced_head_numerator,
                op=MPI.SUM,
                root=0,
            )
            world.Reduce(local_head_area, reduced_head_area, op=MPI.SUM, root=0)
            local_mpi_seconds += time.perf_counter() - mpi_start

            vic_exchange_mm = np.empty(exchange_table.vic_cell_count, dtype=np.float64)
            vic_signed = SignedVolume(0.0, 0.0, 0.0)
            maximum_vic_water_error = None

            if rank == 0:
                assert vic_runtime is not None
                assert reduced_head_numerator is not None
                assert reduced_head_area is not None
                mapped_head = exchange_table.finish_head_mapping(
                    reduced_head_numerator, reduced_head_area
                )

                # write the groundwater state VIC will see for this interval,
                # run exactly one restartable VIC window, and read the signed
                # amount accumulated in OUT_GW_EXCHANGE over that same window.
                prepare_start = time.perf_counter()
                prepared = vic_runtime.prepare_window(
                    window,
                    mapped_head,
                    previous_state=previous_vic_state,
                )
                vic_prepare_seconds = time.perf_counter() - prepare_start
                result = vic_runtime.run_window(prepared)
                previous_vic_state = result.state_file
                vic_seconds = result.spawn_seconds
                io_seconds = result.output_read_seconds
                maximum_vic_water_error = result.maximum_water_error_mm

                map_start = time.perf_counter()
                vic_exchange_mm[:] = exchange_table.extract_vic_values(
                    result.exchange_grid_mm
                )
                vic_signed = exchange_table.source_volume_from_vic_depth(
                    vic_exchange_mm
                )
                local_mapping_seconds += time.perf_counter() - map_start

            # broadcast only active coupled VIC cells rather than a full statewide
            # raster. row/column identity stays in the immutable exchange table,
            # while the time-varying communication payload remains compact.
            mpi_start = time.perf_counter()
            world.Bcast(vic_exchange_mm, root=0)
            local_mpi_seconds += time.perf_counter() - mpi_start

            # convert depth to physical volume before aggregation. conserving
            # q*area instead of averaging depth is what lets the VIC and MF6
            # grids have different cell sizes without changing the water balance.
            map_start = time.perf_counter()
            if rank > 0:
                assert mf6_runtime is not None
                mapping = exchange_table.map_vic_depth_to_model(
                    model_names[rank - 1],
                    vic_exchange_mm,
                    node_count=mf6_runtime.node_count,
                )
                local_mapped = mapping.signed_volume
                local_boundary_target = mapping.node_signed_volume
                local_volume_by_node = mapping.volume_by_node_m3
            else:
                local_mapped = SignedVolume(0.0, 0.0, 0.0)
                local_boundary_target = SignedVolume(0.0, 0.0, 0.0)
                local_volume_by_node = None
            local_mapping_seconds += time.perf_counter() - map_start

            mapped_signed, mpi_elapsed = _reduce_signed_volume(
                world, MPI, local_mapped, rank=rank
            )
            local_mpi_seconds += mpi_elapsed
            boundary_target_signed, mpi_elapsed = _reduce_signed_volume(
                world, MPI, local_boundary_target, rank=rank
            )
            local_mpi_seconds += mpi_elapsed

            if rank == 0:
                assert mapped_signed is not None
                assert boundary_target_signed is not None
                exchange_table.assert_full_mapping_conservation(
                    vic_exchange_mm,
                    mapped_signed,
                    absolute_tolerance_m3=config.coupling.conservation_absolute_tolerance_m3,
                    relative_tolerance=config.coupling.conservation_relative_tolerance,
                )
                # overlap rows preserve gross positive and negative transfer. once
                # those rows are summed onto one API boundary value per MF6 node,
                # opposite-signed overlaps in the same node physically cancel.
                # only the net is invariant across this aggregation step.
                assert_net_volume_close(
                    mapped_signed,
                    boundary_target_signed,
                    absolute_tolerance_m3=config.coupling.conservation_absolute_tolerance_m3,
                    relative_tolerance=config.coupling.conservation_relative_tolerance,
                    label="MF6 node aggregation",
                )
            mpi_start = time.perf_counter()
            world.Barrier()
            local_mpi_seconds += time.perf_counter() - mpi_start

            # apply one constant API rate field to the matching MF6 interval.
            # positive project flux enters groundwater, so API6 receives HCOF=0
            # and RHS=-Q. MF6 is not allowed to step across the coupling boundary.
            local_applied = SignedVolume(0.0, 0.0, 0.0)
            local_api_error = 0.0
            local_iterations = 0
            local_lateral_count = 0
            local_lateral_domain_net = 0.0
            local_lateral_gross = 0.0
            local_lateral_pair_error = 0.0

            if rank > 0:
                assert mf6_runtime is not None
                assert local_volume_by_node is not None
                target_time_days = (
                    window.end - config.mf6.start_time
                ).total_seconds() / 86400.0
                mf6_start = time.perf_counter()
                advance = mf6_runtime.advance_to(
                    target_time_days,
                    local_volume_by_node,
                    api_tolerance_m3_per_day=config.coupling.api_absolute_tolerance_m3_per_day,
                )
                local_mf6_seconds = time.perf_counter() - mf6_start
                local_applied = advance.applied
                local_api_error = advance.maximum_api_error_m3_per_day
                local_iterations = advance.nonlinear_iterations
                if advance.lateral is not None:
                    local_lateral_count = 1
                    local_lateral_domain_net = advance.lateral.domain_net_m3
                    local_lateral_gross = advance.lateral.gross_pair_volume_m3
                    local_lateral_pair_error = (
                        advance.lateral.maximum_pair_antisymmetry_m3
                    )

            applied_signed, mpi_elapsed = _reduce_signed_volume(
                world, MPI, local_applied, rank=rank
            )
            local_mpi_seconds += mpi_elapsed

            mpi_start = time.perf_counter()
            maximum_api_error = world.reduce(local_api_error, op=MPI.MAX, root=0)
            maximum_iterations = world.reduce(local_iterations, op=MPI.MAX, root=0)
            lateral_count = world.reduce(local_lateral_count, op=MPI.SUM, root=0)
            lateral_domain_net = world.reduce(
                local_lateral_domain_net, op=MPI.SUM, root=0
            )
            lateral_gross = world.reduce(local_lateral_gross, op=MPI.SUM, root=0)
            lateral_pair_error = world.reduce(
                local_lateral_pair_error, op=MPI.MAX, root=0
            )
            local_mpi_seconds += time.perf_counter() - mpi_start

            if rank == 0:
                assert boundary_target_signed is not None
                assert applied_signed is not None
                assert_signed_volume_close(
                    boundary_target_signed,
                    applied_signed,
                    absolute_tolerance_m3=config.coupling.conservation_absolute_tolerance_m3,
                    relative_tolerance=config.coupling.conservation_relative_tolerance,
                    label="MF6 API application",
                )

            # reduce head statistics after MF6 advances. only the principal rank
            # reports them, which keeps large MPI jobs readable without hiding
            # rank-specific details from the per-rank log files.
            if rank > 0:
                assert mf6_runtime is not None
                head = mf6_runtime.current_head()
                local_head_min = float(np.min(head))
                local_head_max = float(np.max(head))
                local_head_sum = float(np.sum(head, dtype=np.float64))
                local_head_count = int(head.size)
            else:
                local_head_min = float("inf")
                local_head_max = float("-inf")
                local_head_sum = 0.0
                local_head_count = 0

            mpi_start = time.perf_counter()
            head_min = world.reduce(local_head_min, op=MPI.MIN, root=0)
            head_max = world.reduce(local_head_max, op=MPI.MAX, root=0)
            head_sum = world.reduce(local_head_sum, op=MPI.SUM, root=0)
            head_count = world.reduce(local_head_count, op=MPI.SUM, root=0)
            maximum_mapping_seconds = world.reduce(
                local_mapping_seconds, op=MPI.MAX, root=0
            )
            maximum_mf6_seconds = world.reduce(local_mf6_seconds, op=MPI.MAX, root=0)
            maximum_mpi_seconds = world.reduce(local_mpi_seconds, op=MPI.MAX, root=0)
            local_mpi_seconds += time.perf_counter() - mpi_start

            if rank == 0:
                assert diagnostics is not None
                assert mapped_signed is not None
                assert boundary_target_signed is not None
                assert applied_signed is not None
                assert head_count is not None and head_count > 0
                head_mean = float(head_sum) / int(head_count)
                if lateral_count == len(model_names):
                    reported_lateral_domain_net = float(lateral_domain_net)
                    reported_lateral_gross = float(lateral_gross)
                    reported_lateral_pair_error = float(lateral_pair_error)
                else:
                    reported_lateral_domain_net = None
                    reported_lateral_gross = None
                    reported_lateral_pair_error = None
                    if lateral_count:
                        _log(
                            logger,
                            "warning",
                            "FLOWJA topology was available on only part of the MF6 worker set; whole-domain lateral diagnostics are omitted",
                        )

                window_seconds = time.perf_counter() - window_start_wall
                row = make_window_diagnostics(
                    window=window,
                    vic=vic_signed,
                    mapped=mapped_signed,
                    boundary_target=boundary_target_signed,
                    applied=applied_signed,
                    head_min_m=float(head_min),
                    head_max_m=float(head_max),
                    head_mean_m=head_mean,
                    maximum_api_rate_error_m3_per_day=float(maximum_api_error),
                    maximum_vic_water_error_mm=maximum_vic_water_error,
                    lateral_domain_net_m3=reported_lateral_domain_net,
                    lateral_gross_pair_m3=reported_lateral_gross,
                    lateral_maximum_pair_antisymmetry_m3=reported_lateral_pair_error,
                    nonlinear_iterations_max=int(maximum_iterations),
                    vic_prepare_seconds=vic_prepare_seconds,
                    vic_seconds=vic_seconds,
                    mf6_seconds=float(maximum_mf6_seconds),
                    mapping_seconds=float(maximum_mapping_seconds),
                    mpi_seconds=float(maximum_mpi_seconds),
                    io_seconds=io_seconds,
                    window_seconds=window_seconds,
                )
                diagnostics.append(row)
                _report_window(logger, row, total_windows=len(windows))

        if rank == 0:
            assert diagnostics is not None
            diagnostics.write_summary(
                total_wall_seconds=time.perf_counter() - run_start
            )
        return 0
    finally:
        if mf6_runtime is not None:
            mf6_runtime.finalize()
        if rank > 0 and worker_comm != MPI.COMM_NULL:
            worker_comm.Free()


def _reduce_signed_volume(
    world: Any,
    MPI: Any,
    local: SignedVolume,
    *,
    rank: int,
) -> tuple[SignedVolume | None, float]:
    send = np.asarray(
        [local.positive_m3, local.negative_m3, local.net_m3], dtype=np.float64
    )
    receive = np.empty(3, dtype=np.float64) if rank == 0 else None
    start = time.perf_counter()
    world.Reduce(send, receive, op=MPI.SUM, root=0)
    elapsed = time.perf_counter() - start
    if rank != 0:
        return None, elapsed
    assert receive is not None
    return SignedVolume(
        positive_m3=float(receive[0]),
        negative_m3=float(receive[1]),
        net_m3=float(receive[2]),
    ), elapsed


def _report_window(logger: object, row: Any, *, total_windows: int) -> None:
    _log(logger, "info", "")
    _log(logger, "info", f"coupling step {row.step} / {total_windows}")
    _log(logger, "info", f"vic interval      : {row.start} -> {row.end}")
    _log(
        logger,
        "info",
        f"groundwater head  : min {row.head_min_m:.6f}  max {row.head_max_m:.6f}  mean {row.head_mean_m:.6f} m",
    )
    _log(
        logger,
        "info",
        "vic exchange      : "
        f"positive {row.vic_positive_m3:.6e}  negative {row.vic_negative_m3:.6e}  net {row.vic_net_m3:.6e} m3",
    )
    _log(
        logger,
        "info",
        "mapped overlaps   : "
        f"positive {row.mapped_positive_m3:.6e}  negative {row.mapped_negative_m3:.6e}  net {row.mapped_net_m3:.6e} m3",
    )
    _log(
        logger,
        "info",
        "mf6 boundary      : "
        f"positive {row.boundary_positive_m3:.6e}  negative {row.boundary_negative_m3:.6e}  net {row.boundary_net_m3:.6e} m3",
    )
    _log(
        logger,
        "info",
        f"mf6 solve          : converged, max nonlinear calls {row.nonlinear_iterations_max}",
    )
    _log(
        logger,
        "info",
        f"mapping error      : {row.net_mapping_error_m3:.6e} m3 net",
    )
    _log(
        logger,
        "info",
        f"api error          : {row.net_api_error_m3:.6e} m3 net",
    )
    if row.maximum_vic_water_error_mm is not None:
        _log(
            logger,
            "info",
            f"vic water error    : {row.maximum_vic_water_error_mm:.6e} mm max abs",
        )
    _log(logger, "info", f"elapsed            : {row.window_seconds:.3f} s")


def _log(logger: object, level: str, message: str) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message)
