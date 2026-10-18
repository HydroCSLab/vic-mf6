"""Conservative VIC surface-runoff routing to MODFLOW 6 SFR reaches."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import CouplingRuntimeError
from .records import SignedVolume, VicCell


@dataclass(frozen=True, slots=True)
class RunoffMapping:
    """Window volume assigned to one model's SFR reaches."""

    volume_by_reach_m3: np.ndarray
    signed_volume: SignedVolume


class SurfaceRunoffTable:
    """Immutable VIC-cell to SFR-reach fractions in global VIC-cell order."""

    required_columns = {
        "vic_id",
        "vic_row",
        "vic_col",
        "mf6_model",
        "sfr_package",
        "sfr_reach",
        "weight",
    }

    def __init__(
        self,
        *,
        path: Path,
        vic_cells: tuple[VicCell, ...],
        vic_position: np.ndarray,
        model: np.ndarray,
        package: np.ndarray,
        reach_zero: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        self.path = path
        self.vic_cells = vic_cells
        self.vic_position = np.asarray(vic_position, dtype=np.int64)
        self.model = np.asarray(model, dtype=object)
        self.package = np.asarray(package, dtype=object)
        self.reach_zero = np.asarray(reach_zero, dtype=np.int64)
        self.weight = np.asarray(weight, dtype=np.float64)
        self._area_m2 = np.asarray(
            [cell.area_m2 for cell in vic_cells], dtype=np.float64
        )
        self.model_names = tuple(dict.fromkeys(str(value) for value in self.model))

    @classmethod
    def from_csv(
        cls, path: str | Path, vic_cells: Sequence[VicCell]
    ) -> SurfaceRunoffTable:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise CouplingRuntimeError(f"surface-runoff table was not found: {source}")
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise CouplingRuntimeError("surface-runoff table has no header")
            missing = cls.required_columns - set(reader.fieldnames)
            if missing:
                raise CouplingRuntimeError(
                    "surface-runoff table is missing columns: "
                    + ", ".join(sorted(missing))
                )
            rows = list(reader)
        if not rows:
            raise CouplingRuntimeError("surface-runoff table contains no rows")
        cells = tuple(vic_cells)
        positions = {(cell.vic_id, cell.row, cell.col): cell.position for cell in cells}
        parsed_position = []
        models = []
        packages = []
        reaches = []
        weights = []
        for index, row in enumerate(rows, start=2):
            try:
                key = (str(row["vic_id"]), int(row["vic_row"]), int(row["vic_col"]))
                position = positions[key]
                model = str(row["mf6_model"]).strip().upper()
                package = str(row["sfr_package"]).strip().upper()
                reach = int(row["sfr_reach"])
                weight = float(row["weight"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CouplingRuntimeError(
                    f"invalid surface-runoff relation on CSV line {index}"
                ) from exc
            if not model or not package or reach < 1:
                raise CouplingRuntimeError(
                    f"invalid model, package, or one-based reach on CSV line {index}"
                )
            if not np.isfinite(weight) or weight <= 0.0:
                raise CouplingRuntimeError(
                    f"surface-runoff weight must be finite and positive on CSV line {index}"
                )
            parsed_position.append(position)
            models.append(model)
            packages.append(package)
            reaches.append(reach - 1)
            weights.append(weight)
        parsed_position_array = np.asarray(parsed_position, dtype=np.int64)
        weight_array = np.asarray(weights, dtype=np.float64)
        totals = np.bincount(
            parsed_position_array, weights=weight_array, minlength=len(cells)
        )
        if np.any(np.abs(totals - 1.0) > 1.0e-12):
            bad = np.flatnonzero(np.abs(totals - 1.0) > 1.0e-12)
            raise CouplingRuntimeError(
                "surface-runoff weights must sum to one for every coupled VIC cell; "
                f"bad positions={bad[:20].tolist()}"
            )
        return cls(
            path=source,
            vic_cells=cells,
            vic_position=parsed_position_array,
            model=np.asarray(models, dtype=object),
            package=np.asarray(packages, dtype=object),
            reach_zero=np.asarray(reaches, dtype=np.int64),
            weight=weight_array,
        )

    def for_model(self, model_name: str) -> SurfaceRunoffTable:
        model = str(model_name).upper()
        selected = self.model == model
        if not np.any(selected):
            raise CouplingRuntimeError(
                f"surface-runoff table has no relations for MF6 model {model}"
            )
        return type(self)(
            path=self.path,
            vic_cells=self.vic_cells,
            vic_position=self.vic_position[selected],
            model=self.model[selected],
            package=self.package[selected],
            reach_zero=self.reach_zero[selected],
            weight=self.weight[selected],
        )

    @property
    def package_name(self) -> str:
        names = tuple(dict.fromkeys(str(value) for value in self.package))
        if len(names) != 1:
            raise CouplingRuntimeError(
                "each MF6 model must route runoff to exactly one SFR package"
            )
        return names[0]

    def source_volume(self, runoff_depth_mm: np.ndarray) -> SignedVolume:
        depth = self._depth(runoff_depth_mm)
        return SignedVolume.from_values(depth * 1.0e-3 * self._area_m2)

    def map_to_reaches(self, runoff_depth_mm: np.ndarray) -> RunoffMapping:
        depth = self._depth(runoff_depth_mm)
        row_volume = (
            depth[self.vic_position]
            * 1.0e-3
            * self._area_m2[self.vic_position]
            * self.weight
        )
        volume = np.bincount(
            self.reach_zero,
            weights=row_volume,
            minlength=int(self.reach_zero.max()) + 1,
        ).astype(np.float64, copy=False)
        return RunoffMapping(volume, SignedVolume.from_values(row_volume))

    def _depth(self, values: np.ndarray) -> np.ndarray:
        depth = np.asarray(values, dtype=np.float64).reshape(-1)
        if depth.size != len(self.vic_cells) or not np.all(np.isfinite(depth)):
            raise CouplingRuntimeError(
                "surface-runoff depth must contain one finite value per VIC cell"
            )
        if np.any(depth < -1.0e-10):
            raise CouplingRuntimeError("VIC surface runoff contains negative depth")
        return np.maximum(depth, 0.0)
