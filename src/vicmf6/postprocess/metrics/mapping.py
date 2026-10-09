"""Summarize static overlap geometry without changing mapping weights."""

from __future__ import annotations

from typing import Any

from ...exchange import ExchangeTable


def build_vic_cell_table(
    exchange_table: ExchangeTable,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in exchange_table.vic_cells:
        rows.append(
            {
                "vic_position": cell.position,
                "vic_id": cell.vic_id,
                "vic_row": cell.row,
                "vic_col": cell.col,
                "vic_area_m2": cell.area_m2,
                "vic_lat": cell.latitude,
                "vic_lon": cell.longitude,
                "vic_interface_elevation_m": cell.interface_elevation_m,
            }
        )
    return rows


def build_mapping_tables(
    exchange_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """summarize static VIC and MF6 coupling coverage without losing identities."""

    vic: dict[tuple[str, int, int], dict[str, Any]] = {}
    mf6: dict[tuple[str, int], dict[str, Any]] = {}

    for row in exchange_records:
        vic_key = (str(row["vic_id"]), int(row["vic_row"]), int(row["vic_col"]))
        vic_item = vic.setdefault(
            vic_key,
            {
                "vic_id": vic_key[0],
                "vic_row": vic_key[1],
                "vic_col": vic_key[2],
                "vic_area_m2": row.get("vic_area_m2"),
                "vic_lat": row.get("vic_lat"),
                "vic_lon": row.get("vic_lon"),
                "overlap_count": 0,
                "coupled_area_m2": 0.0,
                "mf6_models": set(),
                "mf6_nodes": set(),
            },
        )
        vic_item["overlap_count"] += 1
        vic_item["coupled_area_m2"] += float(row["overlap_area_m2"])
        vic_item["mf6_models"].add(str(row["mf6_model"]).upper())
        vic_item["mf6_nodes"].add(
            f"{str(row['mf6_model']).upper()}:{int(row['mf6_node'])}"
        )

        mf6_key = (str(row["mf6_model"]).upper(), int(row["mf6_node"]))
        mf6_item = mf6.setdefault(
            mf6_key,
            {
                "mf6_model": mf6_key[0],
                "mf6_node": mf6_key[1],
                "mf6_area_m2": row.get("mf6_area_m2"),
                "overlap_count": 0,
                "coupled_area_m2": 0.0,
                "vic_cells": set(),
            },
        )
        mf6_item["overlap_count"] += 1
        mf6_item["coupled_area_m2"] += float(row["overlap_area_m2"])
        mf6_item["vic_cells"].add(f"{vic_key[0]}@{vic_key[1]},{vic_key[2]}")

    vic_rows: list[dict[str, Any]] = []
    for _, item in sorted(
        vic.items(), key=lambda pair: (pair[0][1], pair[0][2], pair[0][0])
    ):
        area = item["vic_area_m2"]
        item["coverage_fraction"] = (
            None
            if area in (None, 0.0)
            else float(item["coupled_area_m2"]) / float(area)
        )
        item["mf6_models"] = ";".join(sorted(item["mf6_models"]))
        item["mf6_nodes"] = ";".join(sorted(item["mf6_nodes"]))
        vic_rows.append(item)

    mf6_rows: list[dict[str, Any]] = []
    for _, item in sorted(mf6.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        area = item["mf6_area_m2"]
        item["coverage_fraction"] = (
            None
            if area in (None, 0.0)
            else float(item["coupled_area_m2"]) / float(area)
        )
        item["vic_cells"] = ";".join(sorted(item["vic_cells"]))
        mf6_rows.append(item)
    return vic_rows, mf6_rows
