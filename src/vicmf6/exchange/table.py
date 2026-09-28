"""Immutable conservative transfer operator for nonmatching model grids.

Geometry is constructed offline. Runtime work consists only of selecting
known indices, multiplying depth by overlap area, and summing extensive
quantities; identifiers are never inferred from flattened array positions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from ..errors import ExchangeTableError
from .conservation import assert_signed_volume_close
from .loading import read_exchange_table_columns, validate_exchange_records
from .records import HeadContribution, MappingResult, SignedVolume, VicCell
from .validation import (
    _immutable,
    _require_finite,
    _vic_vector,
)


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
        # Integer indices occupy O(overlaps) across all models. A full boolean
        # mask for every model would instead grow as models * overlaps.
        self._model_overlap_indices = {
            model: _immutable(np.flatnonzero(self._overlap_model == model), np.int64)
            for model in self._model_names
        }

        self._model_name_by_casefold = {
            name.casefold(): name for name in self._model_names
        }
        self._vic_areas = _immutable([cell.area_m2 for cell in vic_cells], np.float64)
        self._vic_rows = _immutable([cell.row for cell in vic_cells], np.int64)
        self._vic_cols = _immutable([cell.col for cell in vic_cells], np.int64)
        self._head_area_by_model: dict[str, np.ndarray] = {}

    def for_model(self, model_name: str) -> ExchangeTable:
        """Keep one worker's overlaps while preserving the global VIC cell order.

        MPI broadcasts use the same compact VIC vector on every rank. Workers
        therefore retain VIC identities but need only their own overlap rows.
        """
        model = self._canonical_model_name(model_name)
        indices = self._model_overlap_indices[model]
        return type(self)(
            path=self.path,
            vic_cells=self.vic_cells,
            overlap_vic_position=self._overlap_vic_position[indices],
            overlap_model=self._overlap_model[indices],
            overlap_mf6_node_zero=self._overlap_mf6_node_zero[indices],
            overlap_area_m2=self._overlap_area_m2[indices],
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        require_full_vic_coverage: bool = True,
        coverage_relative_tolerance: float = 1.0e-10,
    ) -> ExchangeTable:
        return cls(
            **read_exchange_table_columns(
                path,
                require_full_vic_coverage=require_full_vic_coverage,
                coverage_relative_tolerance=coverage_relative_tolerance,
            )
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
        return cls(
            **validate_exchange_records(
                records,
                path=path,
                require_full_vic_coverage=require_full_vic_coverage,
                coverage_relative_tolerance=coverage_relative_tolerance,
            )
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
        return self._vic_areas.copy()

    def vic_rows(self) -> np.ndarray:
        return self._vic_rows.copy()

    def vic_cols(self) -> np.ndarray:
        return self._vic_cols.copy()

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
        rows = self._vic_rows
        cols = self._vic_cols
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
        volumes = depth * 1.0e-3 * self._vic_areas
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
        mask = self._model_overlap_indices[model]
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
        mask = self._model_overlap_indices[model]
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
        denominator = self.head_overlap_area_for_model(model)
        return HeadContribution(head_area_sum_m3=numerator, area_sum_m2=denominator)

    def head_overlap_area_for_model(self, model_name: str) -> np.ndarray:
        """Return the static denominator for one worker's reverse head map.

        Cache only models actually used by this rank. In a coupled run a worker
        owns one model, so this adds one VIC-sized vector per worker.
        """
        model = self._canonical_model_name(model_name)
        if model not in self._head_area_by_model:
            indices = self._model_overlap_indices[model]
            self._head_area_by_model[model] = _immutable(
                np.bincount(
                    self._overlap_vic_position[indices],
                    weights=self._overlap_area_m2[indices],
                    minlength=self.vic_cell_count,
                ),
                np.float64,
            )
        return self._head_area_by_model[model]

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
        nodes = self._overlap_mf6_node_zero[self._model_overlap_indices[model]] + 1
        return np.unique(nodes).astype(np.int64, copy=False)

    def model_node_bounds(self, model_name: str) -> tuple[int, int]:
        model = self._canonical_model_name(model_name)
        nodes = self._overlap_mf6_node_zero[self._model_overlap_indices[model]] + 1
        return int(nodes.min()), int(nodes.max())

    def _canonical_model_name(self, model_name: str) -> str:
        requested = str(model_name).strip().casefold()
        model = self._model_name_by_casefold.get(requested)
        if model is None:
            raise ExchangeTableError(
                f"exchange table has no unique model matching {model_name!r}; available={self._model_names}"
            )
        return model
