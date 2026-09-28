"""Physical stages of one explicit window, in the order they are executed.

Heads at the start of the window drive VIC. VIC's accumulated exchange then
becomes a constant groundwater rate over that same window. There is no trial
state, rollback, or iteration between the two models.
"""

from __future__ import annotations

import time

import numpy as np

from ..exchange import SignedVolume, assert_net_volume_close, assert_signed_volume_close
from ..mf6 import Mf6AdvanceResult
from ..schedule import CouplingWindow
from .records import BoundaryExchange, SurfaceExchange, WindowTimings
from .session import CouplingSession

_ZERO_VOLUME = SignedVolume(0.0, 0.0, 0.0)


def map_groundwater_heads_to_vic(
    session: CouplingSession, timings: WindowTimings
) -> np.ndarray | None:
    """Reduce head * overlap area, then divide by the static domain overlap area.

    Summing the numerator before dividing makes the mean independent of how
    groundwater models are partitioned among workers; averaging worker means
    would give small partitions too much weight.
    """
    start = time.perf_counter()
    table = session.exchange_table
    if session.mf6 is not None:
        numerator = table.head_contribution_for_model(
            session.mf6.model_name, session.mf6.current_head()
        ).head_area_sum_m3
    else:
        numerator = np.zeros(table.vic_cell_count, dtype=np.float64)
    timings.mapping_seconds += time.perf_counter() - start
    numerator = session.parallel.reduce_array(numerator, session.parallel.mpi.SUM)
    if not session.parallel.is_controller:
        return None
    assert numerator is not None and session.head_overlap_area_m2 is not None
    return table.finish_head_mapping(numerator, session.head_overlap_area_m2)


def run_vic_and_broadcast_exchange(
    session: CouplingSession,
    window: CouplingWindow,
    mapped_head_m: np.ndarray | None,
    timings: WindowTimings,
) -> SurfaceExchange:
    """Run one restartable VIC window and distribute its integrated depth in mm."""
    depth = np.empty(session.exchange_table.vic_cell_count, dtype=np.float64)
    source_volume = _ZERO_VOLUME
    water_error = None
    if session.vic is not None:
        assert mapped_head_m is not None
        start = time.perf_counter()
        prepared = session.vic.prepare_window(
            window,
            mapped_head_m,
            previous_state=session.previous_vic_state,
        )
        timings.vic_prepare_seconds = time.perf_counter() - start
        result = session.vic.run_window(prepared)
        session.previous_vic_state = result.state_file
        timings.vic_seconds = result.spawn_seconds
        timings.io_seconds = result.output_read_seconds
        water_error = result.maximum_water_error_mm
        start = time.perf_counter()
        depth[:] = session.exchange_table.extract_vic_values(result.exchange_grid_mm)
        source_volume = session.exchange_table.source_volume_from_vic_depth(depth)
        timings.mapping_seconds += time.perf_counter() - start
    session.parallel.broadcast_vic_depth(depth)
    return SurfaceExchange(depth, source_volume, water_error)


def map_exchange_and_check_conservation(
    session: CouplingSession,
    surface: SurfaceExchange,
    timings: WindowTimings,
) -> BoundaryExchange:
    """Convert depth to volume on overlaps, then sum onto groundwater nodes."""
    start = time.perf_counter()
    local_volumes = None
    local_mapped = local_target = _ZERO_VOLUME
    if session.mf6 is not None:
        mapping = session.exchange_table.map_vic_depth_to_model(
            session.mf6.model_name,
            surface.depth_mm,
            node_count=session.mf6.node_count,
        )
        local_volumes = mapping.volume_by_node_m3
        local_mapped = mapping.signed_volume
        local_target = mapping.node_signed_volume
    timings.mapping_seconds += time.perf_counter() - start
    totals = session.parallel.reduce_signed_volumes(local_mapped, local_target)
    mapped = target = None
    if totals is not None:
        mapped, target = totals
        coupling = session.config.coupling
        session.exchange_table.assert_full_mapping_conservation(
            surface.depth_mm,
            mapped,
            absolute_tolerance_m3=coupling.conservation_absolute_tolerance_m3,
            relative_tolerance=coupling.conservation_relative_tolerance,
        )
        # Positive and negative overlap amounts are checked separately above.
        # Opposing amounts can cancel when assigned to the same MF6 node, so
        # only NET volume is invariant across the node-aggregation step.
        assert_net_volume_close(
            mapped,
            target,
            absolute_tolerance_m3=coupling.conservation_absolute_tolerance_m3,
            relative_tolerance=coupling.conservation_relative_tolerance,
            label="MF6 node aggregation",
        )
    session.parallel.wait_for_conservation_check()
    return BoundaryExchange(local_volumes, mapped, target)


def advance_groundwater_and_check_application(
    session: CouplingSession,
    window: CouplingWindow,
    boundary: BoundaryExchange,
    timings: WindowTimings,
) -> tuple[Mf6AdvanceResult | None, SignedVolume | None]:
    """Hold the mapped rate fixed and advance MF6 to the same window boundary.

    MF6 uses its original TDIS substeps. The adapter samples actual API package
    flows after each solve; the domain totals here verify what was applied,
    rather than assuming a requested boundary flux was accepted.
    """
    advance = None
    local_applied = _ZERO_VOLUME
    if session.mf6 is not None:
        assert boundary.volume_by_node_m3 is not None
        target_days = (
            window.end - session.config.mf6.start_time
        ).total_seconds() / 86400.0
        start = time.perf_counter()
        advance = session.mf6.advance_to(
            target_days,
            boundary.volume_by_node_m3,
            api_tolerance_m3_per_day=session.config.coupling.api_absolute_tolerance_m3_per_day,
        )
        timings.mf6_seconds = time.perf_counter() - start
        local_applied = advance.applied
    totals = session.parallel.reduce_signed_volumes(local_applied)
    applied = None
    if totals is not None:
        applied = totals[0]
        assert boundary.target is not None
        coupling = session.config.coupling
        assert_signed_volume_close(
            boundary.target,
            applied,
            absolute_tolerance_m3=coupling.conservation_absolute_tolerance_m3,
            relative_tolerance=coupling.conservation_relative_tolerance,
            label="MF6 API application",
        )
    return advance, applied
