"""Load saved groundwater budget terms and normalize package identifiers.

Budget rates are integrated using their actual native timestep durations.
Missing saved evidence is reported explicitly rather than fabricated as zero."""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ..records import Mf6Geometry
from ..stored_rows import StoredRows
from .mf6_connections import _load_flowja
from .mf6_metadata import (
    _discover_output_path,
    find_model_binary_grid,
    read_budget_step_durations,
)


def load_mf6_budget_data(
    config: ApplicationConfig,
    simulation: Any,
    geometries: dict[str, Mf6Geometry],
    resources: ExitStack,
    scratch_directory: Path,
) -> dict[str, dict[str, Any]]:
    """read CBC terms, API application, and FLOW-JA-FACE when available."""

    try:
        from flopy.utils import CellBudgetFile
    except ImportError as exc:
        raise PostprocessingError("FloPy budget readers are unavailable") from exc

    results: dict[str, dict[str, Any]] = {}
    for source_model in config.mf6_source.models:
        model_name = source_model.name.upper()
        model = simulation.get_model(source_model.name) or simulation.get_model(
            source_model.name.lower()
        )
        cbc_path = (
            _discover_output_path(model, "budget_filerecord", config.mf6.workspace)
            if model is not None
            else None
        )
        if cbc_path is None or not cbc_path.is_file():
            fallback = config.mf6.workspace / f"{source_model.name.lower()}.cbc"
            cbc_path = fallback if fallback.is_file() else cbc_path
        if cbc_path is None or not cbc_path.is_file():
            candidates = sorted(config.mf6.workspace.glob("*.cbc"))
            if len(candidates) == 1:
                cbc_path = candidates[0]
        if cbc_path is None or not cbc_path.is_file():
            results[model_name] = {
                "available": False,
                "reason": "budget file not found",
            }
            continue

        try:
            cbc = CellBudgetFile(str(cbc_path), precision="double")
            resources.callback(cbc.close)
            times = np.asarray(cbc.get_times(), dtype=np.float64)
            names = tuple(
                _decode_budget_name(value) for value in cbc.get_unique_record_names()
            )
        except Exception as exc:
            results[model_name] = {
                "available": False,
                "reason": f"could not read {cbc_path}: {exc}",
            }
            continue

        node_count = int(geometries[model_name].node.size)
        api_name = _select_api_budget_name(names, source_model.api_package)
        flowja_name = _select_budget_name(names, "FLOW-JA-FACE")

        aggregate_terms: list[dict[str, Any]] = []
        try:
            durations = read_budget_step_durations(cbc, times)
            expected = np.asarray(config.mf6_source.time_step_boundaries_days())
            complete = (
                times.size == expected.size
                and times.size > 0
                and np.allclose(times, expected, rtol=0, atol=1e-10)
                and np.allclose(
                    durations, np.diff(np.r_[0.0, expected]), rtol=0, atol=1e-10
                )
            )
            if not complete:
                raise PostprocessingError(
                    "CBC must contain every native TDIS step; use SAVE BUDGET ALL"
                )
            for time_value, dt in zip(times, durations, strict=True):
                for name in names:
                    # DATA-* entries are saved metadata, not water-budget terms.
                    if name == flowja_name or name.startswith("DATA-"):
                        continue
                    vector = _budget_node_vector(
                        cbc, name, float(time_value), node_count
                    )
                    if vector is None:
                        raise PostprocessingError(
                            f"missing budget term {name} at t={time_value}"
                        )
                    positive = float(np.sum(vector[vector > 0.0], dtype=np.float64))
                    negative = float(np.sum(vector[vector < 0.0], dtype=np.float64))
                    net = float(np.sum(vector, dtype=np.float64))
                    aggregate_terms.append(
                        dict(
                            model=model_name,
                            time_days=float(time_value),
                            dt_days=float(dt),
                            term=name,
                            positive_rate_m3_day=positive,
                            negative_rate_m3_day=negative,
                            net_rate_m3_day=net,
                            positive_volume_m3=positive * dt,
                            negative_volume_m3=negative * dt,
                            net_volume_m3=net * dt,
                        )
                    )
        except (PostprocessingError, ValueError, KeyError) as exc:
            results[model_name] = {"available": False, "reason": str(exc)}
            continue

        lateral = _load_flowja(
            find_model_binary_grid(model, config.mf6.workspace),
            cbc,
            flowja_name,
            times,
            node_count,
            durations=durations,
            cells=StoredRows(scratch_directory / f"{model_name}-cells", resources),
            pairs=StoredRows(scratch_directory / f"{model_name}-pairs", resources),
        )
        results[model_name] = {
            "available": True,
            "path": cbc_path,
            "times_days": times,
            "dt_days": durations,
            "complete": True,
            "record_names": names,
            "api_record_name": api_name,
            "read_api_rates": partial(
                _budget_node_vector, cbc, api_name, node_count=node_count
            )
            if api_name
            else None,
            "aggregate_terms": aggregate_terms,
            "lateral": lateral,
        }
    return results


def _decode_budget_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _select_budget_name(names: tuple[str, ...], target: str) -> str | None:
    target_key = target.strip().upper()
    for name in names:
        if name.upper() == target_key:
            return name
    return None


def _select_api_budget_name(names: tuple[str, ...], package_name: str) -> str | None:
    package_key = package_name.strip().upper()
    for name in names:
        if name.upper() == package_key:
            return name
    candidates = [name for name in names if "API" in name.upper()]
    return candidates[0] if len(candidates) == 1 else None


def _budget_node_vector(
    cbc: Any, text: str, totim: float, node_count: int
) -> np.ndarray | None:
    try:
        records = cbc.get_data(text=text, totim=totim)
    except Exception:
        return None
    if not records:
        return None
    total = np.zeros(node_count, dtype=np.float64)
    used = False
    for record in records:
        array = np.asarray(record)
        if array.dtype.names:
            names = {name.lower(): name for name in array.dtype.names}
            q_name = names.get("q")
            node_name = names.get("node") or names.get("node1")
            if q_name is None:
                continue
            q = np.asarray(array[q_name], dtype=np.float64).reshape(-1)
            if node_name is None:
                if q.size != node_count:
                    continue
                total += q
            else:
                nodes = np.asarray(array[node_name], dtype=np.int64).reshape(-1)
                if nodes.size != q.size:
                    continue
                zero = nodes - 1 if nodes.min(initial=1) >= 1 else nodes
                valid = (zero >= 0) & (zero < node_count)
                if not np.all(valid):
                    raise PostprocessingError(
                        f"budget {text} contains out-of-grid node identifiers"
                    )
                np.add.at(total, zero, q)
            used = True
            continue

        flat = np.asarray(array, dtype=np.float64).reshape(-1)
        if flat.size == node_count:
            total += flat
            used = True
    if used and not np.all(np.isfinite(total)):
        raise PostprocessingError(
            f"budget {text} contains non-finite values at t={totim}"
        )
    return total if used else None
