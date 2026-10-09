"""Translate accepted window results into the existing CSV, JSON, and log schema.

Keeping this presentation code out of the algorithm makes additions to reports
independent of model advancement and collective communication.
"""

from logging import Logger
from typing import Any

from ..diagnostics import make_window_diagnostics
from ..exchange import SignedVolume
from ..schedule import CouplingWindow
from .records import (
    BoundaryExchange,
    GroundwaterStatistics,
    SurfaceExchange,
    WindowTimings,
)
from .session import CouplingSession


def write_window_diagnostics(
    session: CouplingSession,
    window: CouplingWindow,
    surface: SurfaceExchange,
    boundary: BoundaryExchange,
    applied: SignedVolume,
    statistics: GroundwaterStatistics,
    timings: WindowTimings,
    *,
    window_seconds: float,
    total_windows: int,
) -> None:
    assert session.diagnostics is not None
    assert boundary.mapped is not None and boundary.target is not None
    lateral_net = lateral_gross = lateral_error = None
    if statistics.lateral_count == len(session.model_names):
        lateral_net = statistics.lateral_domain_net
        lateral_gross = statistics.lateral_gross
        lateral_error = statistics.lateral_pair_error
    elif statistics.lateral_count:
        session.logger.warning(
            "FLOWJA topology was available on only part of the MF6 worker set; whole-domain lateral diagnostics are omitted",
        )
    row = make_window_diagnostics(
        window=window,
        vic=surface.source_volume,
        mapped=boundary.mapped,
        boundary_target=boundary.target,
        applied=applied,
        head_min_m=statistics.head_min,
        head_max_m=statistics.head_max,
        head_mean_m=statistics.head_mean,
        maximum_api_rate_error_m3_per_day=statistics.maximum_api_error,
        maximum_vic_water_error_mm=surface.maximum_water_error_mm,
        lateral_domain_net_m3=lateral_net,
        lateral_gross_pair_m3=lateral_gross,
        lateral_maximum_pair_antisymmetry_m3=lateral_error,
        nonlinear_iterations_max=statistics.maximum_iterations,
        vic_prepare_seconds=timings.vic_prepare_seconds,
        vic_seconds=timings.vic_seconds,
        mf6_seconds=statistics.mf6_seconds,
        mapping_seconds=statistics.mapping_seconds,
        mpi_seconds=statistics.mpi_seconds,
        io_seconds=timings.io_seconds,
        window_seconds=window_seconds,
    )
    session.diagnostics.append(row)
    report_window(session.logger, row, total_windows=total_windows)


def report_window(logger: Logger, row: Any, *, total_windows: int) -> None:
    logger.info("")
    logger.info(f"coupling step {row.step} / {total_windows}")
    logger.info(f"vic interval      : {row.start} -> {row.end}")
    logger.info(
        f"groundwater head  : min {row.head_min_m:.6f}  max {row.head_max_m:.6f}  mean {row.head_mean_m:.6f} m",
    )
    logger.info(
        "vic exchange      : "
        f"positive {row.vic_positive_m3:.6e}  negative {row.vic_negative_m3:.6e}  net {row.vic_net_m3:.6e} m3",
    )
    logger.info(
        "mapped overlaps   : "
        f"positive {row.mapped_positive_m3:.6e}  negative {row.mapped_negative_m3:.6e}  net {row.mapped_net_m3:.6e} m3",
    )
    logger.info(
        "mf6 boundary      : "
        f"positive {row.boundary_positive_m3:.6e}  negative {row.boundary_negative_m3:.6e}  net {row.boundary_net_m3:.6e} m3",
    )
    logger.info(
        f"mf6 solve          : converged, max nonlinear calls {row.nonlinear_iterations_max}",
    )
    logger.info(
        f"mapping error      : {row.net_mapping_error_m3:.6e} m3 net",
    )
    logger.info(
        f"api error          : {row.net_api_error_m3:.6e} m3 net",
    )
    if row.maximum_vic_water_error_mm is not None:
        logger.info(
            f"vic water error    : {row.maximum_vic_water_error_mm:.6e} mm max abs",
        )
    logger.info(f"elapsed            : {row.window_seconds:.3f} s")
