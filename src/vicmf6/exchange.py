"""immutable conservative transfer operator for nonmatching vic and mf6 grids."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ConservationError, ExchangeTableError

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


@dataclass(frozen=True, slots=True)
class SignedVolume:
    """gross positive, gross negative, and signed net transfer in cubic metres."""

    positive_m3: float
    negative_m3: float
    net_m3: float

    @classmethod
    def from_values(cls, values_m3: Sequence[float] | np.ndarray) -> SignedVolume:
        values = np.asarray(values_m3, dtype=np.float64)
        positive = float(values[values > 0.0].sum(dtype=np.float64))
        negative = float(values[values < 0.0].sum(dtype=np.float64))
        return cls(
            positive_m3=positive,
            negative_m3=negative,
            net_m3=float(values.sum(dtype=np.float64)),
        )


@dataclass(frozen=True, slots=True)
class MappingResult:
    """one model's overlap-scale and node-aggregated interface volumes."""

    volume_by_node_m3: np.ndarray
    signed_volume: SignedVolume
    node_signed_volume: SignedVolume


@dataclass(frozen=True, slots=True)
class HeadContribution:
    """partial reverse-map sums to be reduced across groundwater workers."""

    head_area_sum_m3: np.ndarray
    area_sum_m2: np.ndarray


@dataclass(frozen=True, slots=True)
class VicCell:
    """explicit vic array identity used independently from vic_id."""

    position: int
    vic_id: str
    row: int
    col: int
    area_m2: float
    latitude: float | None
    longitude: float | None
    interface_elevation_m: float | None


class ExchangeTable:
    """precomputed geometric overlap table used in both coupling directions.

    runtime mapping is intentionally vectorized. geometry belongs in an offline
    preprocessing step; the time loop only multiplies immutable overlap areas by
    current model state and reduces by explicit cell identifiers.
    """

    def __init__(
        self,
        *,
        path: Path | None,
        vic_cells: tuple[VicCell, ...],
        overlap_vic_position: np.ndarray,
        overlap_model: np.ndarray,
        overlap_mf6_node_zero: np.ndarray,
        overlap_area_m2: np.ndarray,
    ) -> None:
        self.path = path
        self.vic_cells = vic_cells
        self._overlap_vic_position = _immutable(overlap_vic_position, np.int64)
        self._overlap_model = _immutable(overlap_model, object)
        self._overlap_mf6_node_zero = _immutable(overlap_mf6_node_zero, np.int64)
        self._overlap_area_m2 = _immutable(overlap_area_m2, np.float64)
        self._model_names = tuple(
            dict.fromkeys(str(value) for value in self._overlap_model)
        )
        self._model_masks = {
            model: np.asarray(self._overlap_model == model, dtype=bool)
            for model in self._model_names
        }

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        require_full_vic_coverage: bool = True,
        coverage_relative_tolerance: float = 1.0e-10,
    ) -> ExchangeTable:
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

        return cls.from_records(
            records,
            path=source,
            require_full_vic_coverage=require_full_vic_coverage,
            coverage_relative_tolerance=coverage_relative_tolerance,
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, object]],
        *,
        path: Path | None = None,
        require_full_vic_coverage: bool = True,
        coverage_relative_tolerance: float = 1.0e-10,
    ) -> ExchangeTable:
        rows = [dict(record) for record in records]
        if not rows:
            raise ExchangeTableError("exchange table contains no overlap rows")
        if coverage_relative_tolerance < 0.0:
            raise ExchangeTableError("coverage_relative_tolerance must be nonnegative")

        parsed = [_parse_row(row, index) for index, row in enumerate(rows)]
        _reject_duplicate_relations(parsed)

        vic_key_order: list[tuple[str, int, int]] = []
        vic_metadata: dict[
            tuple[str, int, int], dict[str, float | str | int | None]
        ] = {}
        overlap_keys: list[tuple[str, int, int]] = []
        models: list[str] = []
        nodes: list[int] = []
        areas: list[float] = []

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

        return cls(
            path=path,
            vic_cells=tuple(vic_cells),
            overlap_vic_position=overlap_positions,
            overlap_model=np.asarray(models, dtype=object),
            overlap_mf6_node_zero=np.asarray(nodes, dtype=np.int64),
            overlap_area_m2=overlap_area,
        )

    @property
    def model_names(self) -> tuple[str, ...]:
        return self._model_names

    @property
    def overlap_count(self) -> int:
        return int(self._overlap_area_m2.size)

    @property
    def vic_cell_count(self) -> int:
        return len(self.vic_cells)

    @property
    def total_overlap_area_m2(self) -> float:
        return float(self._overlap_area_m2.sum(dtype=np.float64))

    def vic_area_vector_m2(self) -> np.ndarray:
        return np.asarray([cell.area_m2 for cell in self.vic_cells], dtype=np.float64)

    def vic_rows(self) -> np.ndarray:
        return np.asarray([cell.row for cell in self.vic_cells], dtype=np.int64)

    def vic_cols(self) -> np.ndarray:
        return np.asarray([cell.col for cell in self.vic_cells], dtype=np.int64)

    def vic_coordinates(self) -> tuple[np.ndarray, np.ndarray] | None:
        if any(
            cell.latitude is None or cell.longitude is None for cell in self.vic_cells
        ):
            return None
        latitude = np.asarray(
            [cell.latitude for cell in self.vic_cells], dtype=np.float64
        )
        longitude = np.asarray(
            [cell.longitude for cell in self.vic_cells], dtype=np.float64
        )
        return latitude, longitude

    def vic_interface_elevations_m(self) -> np.ndarray:
        if any(cell.interface_elevation_m is None for cell in self.vic_cells):
            raise ExchangeTableError(
                "vic_interface_elevation_m is required for pressure_head_from_interface_elevation"
            )
        return np.asarray(
            [cell.interface_elevation_m for cell in self.vic_cells], dtype=np.float64
        )

    def extract_vic_values(self, full_grid: np.ndarray) -> np.ndarray:
        """extract coupled VIC values by row/column without using vic_id as an array index."""

        grid = np.asarray(full_grid, dtype=np.float64)
        if grid.ndim != 2:
            raise ExchangeTableError(
                f"VIC field must be two-dimensional, got {grid.shape}"
            )
        rows = self.vic_rows()
        cols = self.vic_cols()
        if (
            rows.max(initial=-1) >= grid.shape[0]
            or cols.max(initial=-1) >= grid.shape[1]
        ):
            raise ExchangeTableError(
                f"exchange-table VIC row/column exceeds field shape {grid.shape}"
            )
        values = np.asarray(grid[rows, cols], dtype=np.float64)
        _require_finite(values, "coupled VIC field")
        return values

    def source_volume_from_vic_depth(self, vic_depth_mm: np.ndarray) -> SignedVolume:
        """convert one signed VIC depth amount to full-cell source volumes."""

        depth = _vic_vector(vic_depth_mm, self.vic_cell_count)
        volumes = depth * 1.0e-3 * self.vic_area_vector_m2()
        return SignedVolume.from_values(volumes)

    def overlap_source_volume(self, vic_depth_mm: np.ndarray) -> SignedVolume:
        """compute the signed source represented by overlap rows before MF6 aggregation."""

        depth = _vic_vector(vic_depth_mm, self.vic_cell_count)
        row_volumes = depth[self._overlap_vic_position] * 1.0e-3 * self._overlap_area_m2
        return SignedVolume.from_values(row_volumes)

    def map_vic_depth_to_model(
        self,
        model_name: str,
        vic_depth_mm: np.ndarray,
        *,
        node_count: int | None = None,
    ) -> MappingResult:
        """map signed VIC exchange depth to one model's one-based MF6 nodes."""

        model = self._canonical_model_name(model_name)
        mask = self._model_masks[model]
        depth = _vic_vector(vic_depth_mm, self.vic_cell_count)
        nodes = self._overlap_mf6_node_zero[mask]
        row_volumes = (
            depth[self._overlap_vic_position[mask]]
            * 1.0e-3
            * self._overlap_area_m2[mask]
        )
        minimum_size = int(nodes.max(initial=-1)) + 1
        if node_count is None:
            node_count = minimum_size
        if node_count < minimum_size:
            raise ExchangeTableError(
                f"model {model_name} has coupled mf6_node={minimum_size}, but runtime node_count={node_count}"
            )
        volume_by_node = np.bincount(
            nodes,
            weights=row_volumes,
            minlength=int(node_count),
        ).astype(np.float64, copy=False)
        return MappingResult(
            volume_by_node_m3=volume_by_node,
            signed_volume=SignedVolume.from_values(row_volumes),
            node_signed_volume=SignedVolume.from_values(volume_by_node),
        )

    def head_contribution_for_model(
        self,
        model_name: str,
        head_by_node_m: np.ndarray,
    ) -> HeadContribution:
        """build local area-weighted head sums for one worker before MPI reduction."""

        model = self._canonical_model_name(model_name)
        mask = self._model_masks[model]
        heads = np.asarray(head_by_node_m, dtype=np.float64).reshape(-1)
        nodes = self._overlap_mf6_node_zero[mask]
        if nodes.max(initial=-1) >= heads.size:
            raise ExchangeTableError(
                f"head vector for {model_name} has {heads.size} nodes but exchange table references node {int(nodes.max()) + 1}"
            )
        selected_heads = heads[nodes]
        _require_finite(selected_heads, f"MF6 heads for {model_name}")
        areas = self._overlap_area_m2[mask]
        positions = self._overlap_vic_position[mask]
        numerator = np.bincount(
            positions,
            weights=selected_heads * areas,
            minlength=self.vic_cell_count,
        ).astype(np.float64, copy=False)
        denominator = np.bincount(
            positions,
            weights=areas,
            minlength=self.vic_cell_count,
        ).astype(np.float64, copy=False)
        return HeadContribution(head_area_sum_m3=numerator, area_sum_m2=denominator)

    def finish_head_mapping(
        self,
        head_area_sum_m3: np.ndarray,
        area_sum_m2: np.ndarray,
    ) -> np.ndarray:
        """finish the reverse mapping after all worker contributions are reduced."""

        numerator = _vic_vector(head_area_sum_m3, self.vic_cell_count)
        denominator = _vic_vector(area_sum_m2, self.vic_cell_count)
        if np.any(denominator <= 0.0):
            missing = np.flatnonzero(denominator <= 0.0)
            raise ExchangeTableError(
                f"reverse head map has no groundwater overlap for VIC positions {missing.tolist()}"
            )
        mapped = numerator / denominator
        _require_finite(mapped, "mapped MF6 head")
        return mapped

    def transform_head_for_vic(
        self, absolute_head_m: np.ndarray, mode: str
    ) -> np.ndarray:
        """convert absolute MF6 head to the input convention consumed by the modified VIC."""

        head = _vic_vector(absolute_head_m, self.vic_cell_count)
        normalized = str(mode).strip().lower()
        if normalized == "identity":
            return head.copy()
        if normalized == "pressure_head_from_interface_elevation":
            return head - self.vic_interface_elevations_m()
        raise ExchangeTableError(f"unsupported VIC head transform: {mode}")

    def assert_full_mapping_conservation(
        self,
        vic_depth_mm: np.ndarray,
        mapped_signed_volume: SignedVolume,
        *,
        absolute_tolerance_m3: float,
        relative_tolerance: float,
    ) -> None:
        """check gross positive, gross negative, and net mapped volume independently."""

        expected = self.source_volume_from_vic_depth(vic_depth_mm)
        assert_signed_volume_close(
            expected,
            mapped_signed_volume,
            absolute_tolerance_m3=absolute_tolerance_m3,
            relative_tolerance=relative_tolerance,
            label="VIC-to-MF6 mapping",
        )

    def coupled_nodes(self, model_name: str) -> np.ndarray:
        """return sorted one-based MF6 nodes touched by this model's overlaps."""

        model = self._canonical_model_name(model_name)
        nodes = self._overlap_mf6_node_zero[self._model_masks[model]] + 1
        return np.unique(nodes).astype(np.int64, copy=False)

    def model_node_bounds(self, model_name: str) -> tuple[int, int]:
        model = self._canonical_model_name(model_name)
        nodes = self._overlap_mf6_node_zero[self._model_masks[model]] + 1
        return int(nodes.min()), int(nodes.max())

    def _canonical_model_name(self, model_name: str) -> str:
        requested = str(model_name).strip().casefold()
        matches = [name for name in self._model_names if name.casefold() == requested]
        if len(matches) != 1:
            raise ExchangeTableError(
                f"exchange table has no unique model matching {model_name!r}; available={self._model_names}"
            )
        return matches[0]


def assert_signed_volume_close(
    expected: SignedVolume,
    actual: SignedVolume,
    *,
    absolute_tolerance_m3: float,
    relative_tolerance: float,
    label: str,
) -> None:
    """compare signed volumes without allowing positive and negative errors to cancel."""

    checks = (
        ("positive", expected.positive_m3, actual.positive_m3),
        ("negative", expected.negative_m3, actual.negative_m3),
        ("net", expected.net_m3, actual.net_m3),
    )
    failures: list[str] = []
    for name, expected_value, actual_value in checks:
        error = abs(actual_value - expected_value)
        allowed = absolute_tolerance_m3 + relative_tolerance * max(
            abs(expected_value), abs(actual_value)
        )
        if error > allowed:
            failures.append(
                f"{name}: expected={expected_value:.17g} actual={actual_value:.17g} error={error:.6e} allowed={allowed:.6e}"
            )
    if failures:
        raise ConservationError(
            f"{label} failed signed conservation: " + "; ".join(failures)
        )


def assert_net_volume_close(
    expected: SignedVolume,
    actual: SignedVolume,
    *,
    absolute_tolerance_m3: float,
    relative_tolerance: float,
    label: str,
) -> None:
    """compare only net volume when aggregation can cancel opposite signs."""

    error = abs(actual.net_m3 - expected.net_m3)
    allowed = absolute_tolerance_m3 + relative_tolerance * max(
        abs(expected.net_m3), abs(actual.net_m3)
    )
    if error > allowed:
        raise ConservationError(
            f"{label} failed net conservation: "
            f"expected={expected.net_m3:.17g} actual={actual.net_m3:.17g} "
            f"error={error:.6e} allowed={allowed:.6e}"
        )


def combine_signed_volumes(volumes: Iterable[SignedVolume]) -> SignedVolume:
    items = list(volumes)
    return SignedVolume(
        positive_m3=float(sum(item.positive_m3 for item in items)),
        negative_m3=float(sum(item.negative_m3 for item in items)),
        net_m3=float(sum(item.net_m3 for item in items)),
    )


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
