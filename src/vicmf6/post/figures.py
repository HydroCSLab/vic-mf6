"""publication-oriented figures built only from canonical postprocessing tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import PostprocessingError
from .types import Mf6Geometry, Mf6HeadSeries, PostPaths


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


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PostprocessingError(
            "figure generation requires matplotlib; install vicmf6[post]"
        ) from exc
    return plt


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


def _plot_node_boundary_heatmap(plt: Any, paths: PostPaths, rows, formats, dpi):
    if not rows:
        return []
    models = sorted({str(row["model"]) for row in rows})
    created: list[str] = []
    for model_name in models:
        selected = [row for row in rows if row["model"] == model_name]
        steps = sorted({int(row["step"]) for row in selected})
        nodes = sorted({int(row["node"]) for row in selected})
        matrix = np.full((len(nodes), len(steps)), np.nan, dtype=np.float64)
        step_index = {value: i for i, value in enumerate(steps)}
        node_index = {value: i for i, value in enumerate(nodes)}
        for row in selected:
            matrix[node_index[int(row["node"])], step_index[int(row["step"])]] = float(
                row["target_volume_m3"]
            )
        fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.27 * len(nodes) + 2.0)))
        image = ax.imshow(matrix, aspect="auto", origin="lower")
        fig.colorbar(image, ax=ax, label="Node target volume (m³)")
        ax.set_xlabel("Coupling window")
        ax.set_ylabel("MF6 node")
        ax.set_xticks(np.arange(len(steps)), labels=steps)
        if len(nodes) <= 30:
            ax.set_yticks(np.arange(len(nodes)), labels=nodes)
        ax.set_title(f"{model_name} interface forcing by node")
        created += _save(
            fig,
            plt,
            paths.figures,
            f"07_{model_name.lower()}_node_boundary",
            formats,
            dpi,
        )
    return created


def _plot_vic_exchange_maps(
    plt: Any, paths: PostPaths, exchange_rows, cell_rows, formats, dpi
):
    if not exchange_rows:
        return []
    steps = sorted({int(row["step"]) for row in exchange_rows})
    first = steps[0]
    last = steps[-1]
    created: list[str] = []
    for step, stem, title in (
        (first, "08_vic_exchange_first", f"VIC exchange: window {first}"),
        (last, "09_vic_exchange_last", f"VIC exchange: window {last}"),
    ):
        selected = [row for row in exchange_rows if int(row["step"]) == step]
        created += _plot_vic_field(plt, paths, selected, stem, title, formats, dpi)

    totals: dict[int, float] = defaultdict(float)
    metadata: dict[int, dict[str, Any]] = {}
    for row in exchange_rows:
        position = int(row["vic_position"])
        totals[position] += float(row["exchange_volume_m3"])
        metadata[position] = row
    cumulative = []
    for position in sorted(totals):
        row = dict(metadata[position])
        row["exchange_volume_m3"] = totals[position]
        cumulative.append(row)
    created += _plot_vic_field(
        plt,
        paths,
        cumulative,
        "10_vic_exchange_cumulative",
        "Cumulative VIC exchange volume",
        formats,
        dpi,
    )
    return created


def _plot_vic_field(plt: Any, paths: PostPaths, rows, stem, title, formats, dpi):
    if not rows:
        return []
    max_row = max(int(row["vic_row"]) for row in rows)
    max_col = max(int(row["vic_col"]) for row in rows)
    matrix = np.full((max_row + 1, max_col + 1), np.nan, dtype=np.float64)
    for row in rows:
        matrix[int(row["vic_row"]), int(row["vic_col"])] = float(
            row["exchange_volume_m3"]
        )
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    image = ax.imshow(matrix, origin="upper", aspect="equal")
    fig.colorbar(image, ax=ax, label="Exchange volume (m³)")
    ax.set_xlabel("VIC column")
    ax.set_ylabel("VIC row")
    ax.set_title(title)
    return _save(fig, plt, paths.figures, stem, formats, dpi)


def _plot_mf6_head_maps(
    plt: Any, paths: PostPaths, head_series, geometries, formats, dpi
):
    created: list[str] = []
    for model_name in sorted(head_series):
        series = head_series[model_name]
        geometry = geometries[model_name]
        values = (
            (
                series.initial_heads_m,
                "11",
                "initial_head",
                "Initial hydraulic head",
                "Hydraulic head (m)",
            ),
            (
                series.heads_m[-1],
                "12",
                "final_head",
                "Final hydraulic head",
                "Hydraulic head (m)",
            ),
            (
                series.heads_m[-1] - series.initial_heads_m,
                "13",
                "head_change",
                "Hydraulic head change",
                "Head change (m)",
            ),
        )
        for field, order, label, title, colorbar in values:
            created += _plot_mf6_field(
                plt,
                paths,
                model_name,
                geometry,
                np.asarray(field, dtype=np.float64),
                f"{order}_{model_name.lower()}_{label}",
                f"{model_name}: {title}",
                colorbar,
                formats,
                dpi,
            )
    return created


def _plot_mf6_field(
    plt: Any,
    paths: PostPaths,
    model_name,
    geometry,
    values,
    stem,
    title,
    colorbar,
    formats,
    dpi,
):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    if geometry.row is not None and geometry.col is not None:
        matrix = np.full(
            (int(np.max(geometry.row)) + 1, int(np.max(geometry.col)) + 1), np.nan
        )
        for index, value in enumerate(values):
            matrix[int(geometry.row[index]), int(geometry.col[index])] = float(value)
        artist = ax.imshow(matrix, origin="upper", aspect="equal")
        ax.set_xlabel("MF6 column")
        ax.set_ylabel("MF6 row")
    elif geometry.vertices is not None:
        try:
            from matplotlib.collections import PolyCollection
        except ImportError as exc:
            raise PostprocessingError(
                "matplotlib PolyCollection is unavailable"
            ) from exc
        polygons = [list(vertices) for vertices in geometry.vertices]
        artist = PolyCollection(
            polygons, array=np.asarray(values), edgecolors="black", linewidths=0.4
        )
        ax.add_collection(artist)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif geometry.x is not None and geometry.y is not None:
        artist = ax.scatter(geometry.x, geometry.y, c=values, s=75)
        ax.set_xlabel("x / longitude")
        ax.set_ylabel("y / latitude")
    else:
        artist = ax.scatter(
            np.arange(1, values.size + 1), np.zeros(values.size), c=values, s=75
        )
        ax.set_xlabel("MF6 node")
        ax.set_yticks([])
    fig.colorbar(artist, ax=ax, label=colorbar)
    ax.set_title(title)
    return _save(fig, plt, paths.figures, stem, formats, dpi)


def _plot_mapping_connectivity(
    plt: Any, paths: PostPaths, records, geometries, formats, dpi
):
    if not records:
        return []
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    vic_points: dict[tuple[str, int, int], tuple[float, float]] = {}
    for row in records:
        if row.get("vic_lon") is None or row.get("vic_lat") is None:
            continue
        vic_points[(str(row["vic_id"]), int(row["vic_row"]), int(row["vic_col"]))] = (
            float(row["vic_lon"]),
            float(row["vic_lat"]),
        )
    if not vic_points:
        plt.close(fig)
        return []

    vx = [point[0] for point in vic_points.values()]
    vy = [point[1] for point in vic_points.values()]
    ax.scatter(vx, vy, marker="s", s=55, label="VIC cells")

    for model_name, geometry in geometries.items():
        if geometry.x is None or geometry.y is None:
            continue
        ax.scatter(
            geometry.x, geometry.y, marker="o", s=55, label=f"{model_name} cells"
        )
        for row in records:
            if str(row["mf6_model"]).upper() != model_name:
                continue
            key = (str(row["vic_id"]), int(row["vic_row"]), int(row["vic_col"]))
            point = vic_points.get(key)
            node = int(row["mf6_node"]) - 1
            if point is None or node < 0 or node >= geometry.x.size:
                continue
            ax.plot(
                [point[0], float(geometry.x[node])],
                [point[1], float(geometry.y[node])],
                linewidth=0.45,
                alpha=0.35,
            )
    ax.set_xlabel("Longitude / x")
    ax.set_ylabel("Latitude / y")
    ax.set_title("VIC–MF6 overlap connectivity")
    ax.legend()
    ax.grid(True, alpha=0.2)
    return _save(fig, plt, paths.figures, "14_mapping_connectivity", formats, dpi)


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


def _save(
    fig: Any, plt: Any, directory: Path, stem: str, formats: Iterable[str], dpi: int
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    created: list[str] = []
    for extension in formats:
        normalized = extension.strip().lower()
        if normalized not in {"png", "pdf", "svg"}:
            raise PostprocessingError(f"unsupported figure format: {extension}")
        path = directory / f"{stem}.{normalized}"
        kwargs = {"bbox_inches": "tight"}
        if normalized == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        created.append(path.name)
    plt.close(fig)
    return created
