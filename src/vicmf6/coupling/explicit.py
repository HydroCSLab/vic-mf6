"""The complete explicit VIC–MF6 algorithm, without native or MPI bookkeeping."""

from __future__ import annotations

import time

from ..config import ApplicationConfig
from ..errors import CouplingRuntimeError
from ..schedule import build_windows
from .records import WindowTimings
from .reporting import write_window_diagnostics
from .session import CouplingSession
from .stages import (
    advance_groundwater_and_check_application,
    map_exchange_and_check_conservation,
    map_groundwater_heads_to_vic,
    run_vic_and_broadcast_exchange,
)


def run_coupling(config: ApplicationConfig, *, logger: object) -> int:
    """Advance both models once per window using heads from the previous window.

    Run on every outer MPI rank. A rank-local exception propagates to the CLI,
    which aborts the world before peers can hang in collective finalization.
    """
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise CouplingRuntimeError(
            "mpi4py is required for coupled execution; install vicmf6[runtime]"
        ) from exc

    session = CouplingSession.initialize(config, logger, MPI)
    windows = build_windows(
        config.coupling.start_time,
        config.coupling.end_time,
        config.coupling.interval_days,
    )
    run_start = time.perf_counter()
    for window in windows:
        window_start = time.perf_counter()
        timings = WindowTimings()
        session.parallel.elapsed_seconds = 0.0

        # 1. Give VIC the groundwater head at the beginning of this window.
        mapped_head = map_groundwater_heads_to_vic(session, timings)
        # 2. Advance VIC once and read its signed accumulated exchange depth.
        surface = run_vic_and_broadcast_exchange(session, window, mapped_head, timings)
        # 3. Convert mm to m3 and verify conservation before solving groundwater.
        boundary = map_exchange_and_check_conservation(session, surface, timings)
        # 4. Advance persistent MF6 models under that fixed exchange rate.
        advance, applied = advance_groundwater_and_check_application(
            session, window, boundary, timings
        )
        # 5. Record accepted results; these heads will drive the next VIC window.
        statistics = session.parallel.reduce_groundwater_statistics(advance, timings)
        if session.parallel.is_controller:
            assert statistics is not None and applied is not None
            write_window_diagnostics(
                session,
                window,
                surface,
                boundary,
                applied,
                statistics,
                timings,
                window_seconds=time.perf_counter() - window_start,
                total_windows=len(windows),
            )

    if session.diagnostics is not None:
        session.diagnostics.write_summary(
            total_wall_seconds=time.perf_counter() - run_start
        )
    session.finalize_successful_run()
    return 0
