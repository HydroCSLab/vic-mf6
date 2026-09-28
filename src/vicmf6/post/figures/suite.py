"""Select the standard diagnostic figure suite from canonical report tables."""

from __future__ import annotations

from typing import Any

from ..types import Mf6Geometry, Mf6HeadSeries, PostPaths
from .output import _matplotlib
from .spatial import (
    _plot_mapping_connectivity,
    _plot_mf6_head_maps,
    _plot_node_boundary_heatmap,
    _plot_vic_exchange_maps,
)
from .time_series import (
    _plot_conservation_errors,
    _plot_cumulative_exchange,
    _plot_daily_net_exchange,
    _plot_daily_signed_exchange,
    _plot_head_statistics,
    _plot_lateral_flow,
    _plot_node_head_timeseries,
    _plot_runtime,
)


def create_all_figures(
    paths: PostPaths,
    *,
    window_rows: list[dict[str, Any]],
    vic_exchange_rows: list[dict[str, Any]],
    vic_cell_rows: list[dict[str, Any]],
    node_boundary_rows: list[dict[str, Any]],
    head_series: dict[str, Mf6HeadSeries],
    geometries: dict[str, Mf6Geometry],
    exchange_records: list[dict[str, Any]],
    lateral_pair_rows: list[dict[str, Any]],
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 220,
) -> list[str]:
    """write the standard diagnostic figure suite and return created file names."""

    plt = _matplotlib()
    created: list[str] = []

    created += _plot_daily_net_exchange(plt, paths, window_rows, formats, dpi)
    created += _plot_daily_signed_exchange(plt, paths, window_rows, formats, dpi)
    created += _plot_cumulative_exchange(plt, paths, window_rows, formats, dpi)
    created += _plot_head_statistics(plt, paths, head_series, formats, dpi)
    created += _plot_conservation_errors(plt, paths, window_rows, formats, dpi)
    created += _plot_runtime(plt, paths, window_rows, formats, dpi)
    created += _plot_node_boundary_heatmap(plt, paths, node_boundary_rows, formats, dpi)
    created += _plot_vic_exchange_maps(
        plt, paths, vic_exchange_rows, vic_cell_rows, formats, dpi
    )
    created += _plot_mf6_head_maps(plt, paths, head_series, geometries, formats, dpi)
    created += _plot_mapping_connectivity(
        plt, paths, exchange_records, geometries, formats, dpi
    )
    created += _plot_node_head_timeseries(plt, paths, head_series, formats, dpi)
    if lateral_pair_rows:
        created += _plot_lateral_flow(plt, paths, lateral_pair_rows, formats, dpi)
    return created
