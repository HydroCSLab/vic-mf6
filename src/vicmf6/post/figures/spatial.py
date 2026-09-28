"""Plot cell identity, mapped transfer, and groundwater spatial fields."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ...errors import PostprocessingError
from ..records import PostPaths
from .output import _save


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
