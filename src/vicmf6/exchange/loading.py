"""Read and validate static overlap geometry before any native model starts.

CSV parsing, identity checks, and coverage checks belong here. The transfer
operator receives validated columns and never reparses geometry in a timestep.
One-based MF6 node numbers are converted exactly once at this boundary.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from ..errors import ExchangeTableError
from .records import VicCell
from .validation import (
    _REQUIRED_COLUMNS,
    _merge_vic_metadata,
    _optional_float_value,
    _parse_row,
    _reject_duplicate_relations,
    _vic_metadata_from_row,
)


def read_exchange_table_columns(
    path: str | Path,
    *,
    require_full_vic_coverage: bool = True,
    coverage_relative_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ExchangeTableError(f"exchange table was not found: {source}")

    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ExchangeTableError(f"exchange table has no header: {source}")
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ExchangeTableError(
                "exchange table is missing required columns: "
                + ", ".join(sorted(missing))
            )
        records = list(reader)

    return validate_exchange_records(
        records,
        path=source,
        require_full_vic_coverage=require_full_vic_coverage,
        coverage_relative_tolerance=coverage_relative_tolerance,
    )


def validate_exchange_records(
    records: Iterable[Mapping[str, object]],
    *,
    path: Path | None = None,
    require_full_vic_coverage: bool = True,
    coverage_relative_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    rows = [dict(record) for record in records]
    if not rows:
        raise ExchangeTableError("exchange table contains no overlap rows")
    if (
        not np.isfinite(coverage_relative_tolerance)
        or coverage_relative_tolerance < 0.0
    ):
        raise ExchangeTableError("coverage_relative_tolerance must be nonnegative")

    parsed = [_parse_row(row, index) for index, row in enumerate(rows)]
    _reject_duplicate_relations(parsed)
    coordinates_by_id = {}
    ids_by_coordinates = {}
    for row in parsed:
        cell_id = str(row["vic_id"])
        coordinates = (int(row["vic_row"]), int(row["vic_col"]))
        if (
            coordinates_by_id.setdefault(cell_id, coordinates) != coordinates
            or ids_by_coordinates.setdefault(coordinates, cell_id) != cell_id
        ):
            raise ExchangeTableError(
                "each VIC cell must have one unique vic_id and row/column pair"
            )

    vic_key_order: list[tuple[str, int, int]] = []
    vic_metadata: dict[tuple[str, int, int], dict[str, float | str | int | None]] = {}
    overlap_keys: list[tuple[str, int, int]] = []
    models: list[str] = []
    nodes: list[int] = []
    areas: list[float] = []
    mf6_interface_elevations: list[float] = []

    for row in parsed:
        key = (str(row["vic_id"]), int(row["vic_row"]), int(row["vic_col"]))
        if key not in vic_metadata:
            vic_key_order.append(key)
            vic_metadata[key] = _vic_metadata_from_row(row)
        else:
            _merge_vic_metadata(vic_metadata[key], row, key)

        overlap_keys.append(key)
        models.append(str(row["mf6_model"]))
        nodes.append(int(row["mf6_node"]) - 1)
        areas.append(float(row["overlap_area_m2"]))
        elevation = row.get("mf6_interface_elevation_m")
        mf6_interface_elevations.append(
            np.nan if elevation is None else float(elevation)
        )

    vic_position_by_key = {key: index for index, key in enumerate(vic_key_order)}
    overlap_positions = np.asarray(
        [vic_position_by_key[key] for key in overlap_keys], dtype=np.int64
    )
    overlap_area = np.asarray(areas, dtype=np.float64)
    coverage = np.bincount(
        overlap_positions,
        weights=overlap_area,
        minlength=len(vic_key_order),
    ).astype(np.float64, copy=False)

    vic_cells: list[VicCell] = []
    for position, key in enumerate(vic_key_order):
        metadata = vic_metadata[key]
        declared_area = metadata["vic_area_m2"]
        if declared_area is None:
            if require_full_vic_coverage:
                raise ExchangeTableError(
                    "vic_area_m2 is required when full VIC coverage is enforced; "
                    f"missing for vic_id={key[0]} row={key[1]} col={key[2]}"
                )
            declared_area = float(coverage[position])

        declared_area_float = float(declared_area)
        if declared_area_float <= 0.0 or not np.isfinite(declared_area_float):
            raise ExchangeTableError(f"invalid vic_area_m2 for VIC cell {key}")

        if require_full_vic_coverage:
            error = abs(float(coverage[position]) - declared_area_float)
            allowed = coverage_relative_tolerance * max(declared_area_float, 1.0)
            if error > allowed:
                raise ExchangeTableError(
                    "overlap area does not reproduce the declared VIC cell area: "
                    f"vic_id={key[0]} row={key[1]} col={key[2]} "
                    f"overlap={coverage[position]:.17g} vic_area={declared_area_float:.17g} "
                    f"error={error:.6e} allowed={allowed:.6e}"
                )

        vic_cells.append(
            VicCell(
                position=position,
                vic_id=key[0],
                row=key[1],
                col=key[2],
                area_m2=declared_area_float,
                latitude=_optional_float_value(metadata["vic_lat"]),
                longitude=_optional_float_value(metadata["vic_lon"]),
                interface_elevation_m=_optional_float_value(
                    metadata["vic_interface_elevation_m"]
                ),
            )
        )

    return dict(
        path=path,
        vic_cells=tuple(vic_cells),
        overlap_vic_position=overlap_positions,
        overlap_model=np.asarray(models, dtype=object),
        overlap_mf6_node_zero=np.asarray(nodes, dtype=np.int64),
        overlap_area_m2=overlap_area,
        overlap_mf6_interface_elevation_m=np.asarray(
            mf6_interface_elevations, dtype=np.float64
        ),
    )
