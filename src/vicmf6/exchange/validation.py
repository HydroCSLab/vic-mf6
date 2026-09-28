"""Validate overlap identities, geometry, and finite runtime fields.

Reject conflicting metadata rather than guessing a cell or normalizing a bad
area. Such repairs can create plausible outputs while changing the water budget."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..errors import ExchangeTableError

_REQUIRED_COLUMNS = {
    "vic_id",
    "vic_row",
    "vic_col",
    "mf6_model",
    "mf6_node",
    "overlap_area_m2",
}


_OPTIONAL_FLOAT_COLUMNS = {
    "vic_area_m2",
    "mf6_area_m2",
    "vic_lat",
    "vic_lon",
    "vic_interface_elevation_m",
}


def _parse_row(raw: Mapping[str, object], index: int) -> dict[str, object]:
    missing = _REQUIRED_COLUMNS - set(raw)
    if missing:
        raise ExchangeTableError(
            f"exchange row {index + 2} is missing columns: {', '.join(sorted(missing))}"
        )

    row: dict[str, object] = {
        "vic_id": str(raw["vic_id"]).strip(),
        "vic_row": _integer(raw["vic_row"], "vic_row", index, minimum=0),
        "vic_col": _integer(raw["vic_col"], "vic_col", index, minimum=0),
        "mf6_model": str(raw["mf6_model"]).strip(),
        "mf6_node": _integer(raw["mf6_node"], "mf6_node", index, minimum=1),
        "overlap_area_m2": _finite_float(
            raw["overlap_area_m2"], "overlap_area_m2", index, positive=True
        ),
    }
    if not row["vic_id"]:
        raise ExchangeTableError(f"exchange row {index + 2} has an empty vic_id")
    if not row["mf6_model"]:
        raise ExchangeTableError(f"exchange row {index + 2} has an empty mf6_model")

    for column in _OPTIONAL_FLOAT_COLUMNS:
        value = raw.get(column)
        row[column] = (
            None
            if value is None or str(value).strip() == ""
            else _finite_float(
                value,
                column,
                index,
                positive=column in {"vic_area_m2", "mf6_area_m2"},
            )
        )
    return row


def _vic_metadata_from_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "vic_id": row["vic_id"],
        "vic_row": row["vic_row"],
        "vic_col": row["vic_col"],
        "vic_area_m2": row.get("vic_area_m2"),
        "vic_lat": row.get("vic_lat"),
        "vic_lon": row.get("vic_lon"),
        "vic_interface_elevation_m": row.get("vic_interface_elevation_m"),
    }


def _merge_vic_metadata(
    stored: dict[str, object], row: Mapping[str, object], key: tuple[str, int, int]
) -> None:
    for column in ("vic_area_m2", "vic_lat", "vic_lon", "vic_interface_elevation_m"):
        incoming = row.get(column)
        if incoming is None:
            continue
        existing = stored.get(column)
        if existing is None:
            stored[column] = incoming
            continue
        if not np.isclose(float(existing), float(incoming), rtol=0.0, atol=1.0e-12):
            raise ExchangeTableError(
                f"inconsistent {column} across overlap rows for VIC cell {key}: {existing} vs {incoming}"
            )


def _reject_duplicate_relations(rows: Sequence[Mapping[str, object]]) -> None:
    seen: set[tuple[str, int, int, str, int]] = set()
    for index, row in enumerate(rows):
        key = (
            str(row["vic_id"]),
            int(row["vic_row"]),
            int(row["vic_col"]),
            str(row["mf6_model"]).casefold(),
            int(row["mf6_node"]),
        )
        if key in seen:
            raise ExchangeTableError(
                f"duplicate VIC/MF6 overlap relationship at exchange row {index + 2}: {key}"
            )
        seen.add(key)


def _vic_vector(values: np.ndarray, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != expected_size:
        raise ExchangeTableError(
            f"VIC vector size must be {expected_size}, got {array.size}"
        )
    _require_finite(array, "VIC vector")
    return array


def _integer(raw: object, field: str, row_index: int, minimum: int) -> int:
    try:
        value_float = float(raw)
        value = int(value_float)
    except (TypeError, ValueError) as exc:
        raise ExchangeTableError(
            f"exchange row {row_index + 2} has invalid {field}: {raw!r}"
        ) from exc
    if value_float != value or value < minimum:
        raise ExchangeTableError(
            f"exchange row {row_index + 2} has invalid {field}: {raw!r}"
        )
    return value


def _finite_float(
    raw: object,
    field: str,
    row_index: int,
    *,
    positive: bool = False,
) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ExchangeTableError(
            f"exchange row {row_index + 2} has invalid {field}: {raw!r}"
        ) from exc
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ExchangeTableError(
            f"exchange row {row_index + 2} has invalid {field}: {raw!r}"
        )
    return value


def _optional_float_value(value: object) -> float | None:
    return None if value is None else float(value)


def _require_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        bad = np.flatnonzero(~np.isfinite(values))
        raise ExchangeTableError(
            f"{label} contains non-finite values at positions {bad[:20].tolist()}"
        )


def _immutable(values: np.ndarray, dtype: object) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array
