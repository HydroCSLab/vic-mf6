"""Plot temporal diagnostics without recomputing the coupling physics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..types import PostPaths
from .output import _save


def _plot_daily_net_exchange(
    plt: Any, paths: PostPaths, rows: list[dict[str, Any]], formats, dpi
):
    x = np.array([int(row["step"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        x, [float(row["vic_net_m3"]) for row in rows], marker="o", label="VIC exchange"
    )
    ax.plot(
        x,
        [float(row["boundary_net_m3"]) for row in rows],
        marker="s",
        label="MF6 node target",
    )
    ax.plot(
        x,
        [float(row["applied_net_m3"]) for row in rows],
        marker="^",
        label="MF6 API applied",
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Coupling window")
    ax.set_ylabel("Net exchange volume (m³)")
    ax.set_title("Daily coupled exchange")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return _save(fig, plt, paths.figures, "01_daily_net_exchange", formats, dpi)


def _plot_daily_signed_exchange(
    plt: Any, paths: PostPaths, rows: list[dict[str, Any]], formats, dpi
):
    x = np.array([int(row["step"]) for row in rows])
    positive = np.asarray(
        [float(row["vic_positive_m3"]) for row in rows], dtype=np.float64
    )
    upward = np.asarray(
        [-float(row["vic_negative_m3"]) for row in rows], dtype=np.float64
    )

    # Plot the two directions on separate axes as positive magnitudes.  A common
    # signed axis can visually hide the smaller branch when one direction is an
    # order of magnitude larger, even though the run is genuinely bidirectional.
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    axes[0].bar(x, positive)
    axes[0].set_ylabel("VIC→MF6 (m³)")
    axes[0].set_title("Gross exchange by direction")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, upward)
    axes[1].set_xlabel("Coupling window")
    axes[1].set_ylabel("MF6→VIC (m³)")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.text(
        0.99,
        0.01,
        f"cumulative: VIC→MF6={positive.sum():.6g} m³; MF6→VIC={upward.sum():.6g} m³",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    return _save(fig, plt, paths.figures, "02_daily_signed_exchange", formats, dpi)


def _plot_cumulative_exchange(
    plt: Any, paths: PostPaths, rows: list[dict[str, Any]], formats, dpi
):
    x = np.array([int(row["step"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for key, label in (
        ("vic_net_m3", "VIC"),
        ("boundary_net_m3", "MF6 node target"),
        ("applied_net_m3", "MF6 API applied"),
    ):
        cumulative = np.cumsum([float(row[key]) for row in rows], dtype=np.float64)
        ax.plot(x, cumulative, marker="o", label=label)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Coupling window")
    ax.set_ylabel("Cumulative net exchange (m³)")
    ax.set_title("Cumulative coupled exchange")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return _save(fig, plt, paths.figures, "03_cumulative_net_exchange", formats, dpi)


def _plot_head_statistics(plt: Any, paths: PostPaths, head_series, formats, dpi):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for model_name in sorted(head_series):
        series = head_series[model_name]
        minimum = np.min(series.heads_m, axis=1)
        mean = np.mean(series.heads_m, axis=1)
        maximum = np.max(series.heads_m, axis=1)
        ax.plot(series.times_days, minimum, marker="o", label=f"{model_name} min")
        ax.plot(series.times_days, mean, marker="s", label=f"{model_name} mean")
        ax.plot(series.times_days, maximum, marker="^", label=f"{model_name} max")
    ax.set_xlabel("MF6 elapsed time (days)")
    ax.set_ylabel("Hydraulic head (m)")
    ax.set_title("Groundwater head statistics")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    return _save(fig, plt, paths.figures, "04_mf6_head_statistics", formats, dpi)


def _plot_conservation_errors(plt: Any, paths: PostPaths, rows, formats, dpi):
    x = np.array([int(row["step"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    series = (
        ("net_mapping_error_m3", "VIC→overlap net"),
        ("aggregation_net_error_m3", "overlap→node net"),
        ("net_api_error_m3", "node→API net"),
    )
    floor = np.finfo(np.float64).tiny
    for key, label in series:
        values = np.maximum(np.abs([float(row[key]) for row in rows]), floor)
        ax.semilogy(x, values, marker="o", label=label)
    ax.set_xlabel("Coupling window")
    ax.set_ylabel("Absolute conservation error (m³)")
    ax.set_title("Cross-model conservation diagnostics")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    return _save(fig, plt, paths.figures, "05_conservation_errors", formats, dpi)


def _plot_runtime(plt: Any, paths: PostPaths, rows, formats, dpi):
    x = np.array([int(row["step"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for key, label in (
        ("vic_seconds", "VIC"),
        ("mf6_seconds", "MF6"),
        ("mapping_seconds", "mapping"),
        ("mpi_seconds", "MPI"),
        ("io_seconds", "I/O"),
    ):
        ax.plot(x, [float(row[key]) for row in rows], marker="o", label=label)
    ax.set_xlabel("Coupling window")
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-window runtime components")
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.25)
    return _save(fig, plt, paths.figures, "06_runtime_components", formats, dpi)


def _plot_node_head_timeseries(plt: Any, paths: PostPaths, head_series, formats, dpi):
    created: list[str] = []
    for model_name, series in sorted(head_series.items()):
        if series.node_count > 32:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for node in range(series.node_count):
            ax.plot(
                series.times_days,
                series.heads_m[:, node],
                linewidth=1.0,
                label=f"node {node + 1}",
            )
        ax.set_xlabel("MF6 elapsed time (days)")
        ax.set_ylabel("Hydraulic head (m)")
        ax.set_title(f"{model_name} node head trajectories")
        if series.node_count <= 16:
            ax.legend(ncol=3, fontsize="small")
        ax.grid(True, alpha=0.25)
        created += _save(
            fig, plt, paths.figures, f"15_{model_name.lower()}_node_heads", formats, dpi
        )
    return created


def _plot_lateral_flow(plt: Any, paths: PostPaths, rows, formats, dpi):
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["time_days"])].append(row)
    times = sorted(grouped)
    maximum_pair = [
        max(float(item["pair_magnitude_m3_day"]) for item in grouped[t]) for t in times
    ]
    antisymmetry = [
        max(abs(float(item["antisymmetry_error_m3_day"])) for item in grouped[t])
        for t in times
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(times, maximum_pair, marker="o", label="maximum lateral pair magnitude")
    ax.set_xlabel("MF6 elapsed time (days)")
    ax.set_ylabel("FLOW-JA-FACE magnitude (m³/day)")
    ax.set_title("Connected groundwater lateral-flow strength")
    ax.grid(True, alpha=0.25)
    first = _save(fig, plt, paths.figures, "16_lateral_flow_magnitude", formats, dpi)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    floor = np.finfo(np.float64).tiny
    ax.semilogy(times, np.maximum(antisymmetry, floor), marker="o")
    ax.set_xlabel("MF6 elapsed time (days)")
    ax.set_ylabel("Pair antisymmetry error (m³/day)")
    ax.set_title("FLOW-JA-FACE pair antisymmetry")
    ax.grid(True, which="both", alpha=0.25)
    second = _save(
        fig, plt, paths.figures, "17_lateral_pair_antisymmetry", formats, dpi
    )
    return first + second
