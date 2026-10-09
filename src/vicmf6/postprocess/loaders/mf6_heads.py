"""Read initial and saved groundwater heads in model-node order."""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ..records import Mf6HeadSeries
from .mf6_metadata import _discover_output_path


def load_mf6_head_series(
    config: ApplicationConfig,
    simulation: Any,
    resources: ExitStack,
) -> dict[str, Mf6HeadSeries]:
    """Index saved heads; read one record at a time until resources are closed."""

    try:
        from flopy.utils import HeadFile, HeadUFile
    except ImportError as exc:
        raise PostprocessingError("FloPy head readers are unavailable") from exc

    result: dict[str, Mf6HeadSeries] = {}
    for source_model in config.mf6_source.models:
        model_name = source_model.name
        model = simulation.get_model(model_name)
        if model is None:
            model = simulation.get_model(model_name.lower())
        if model is None:
            raise PostprocessingError(f"FloPy did not load GWF model {model_name}")
        if getattr(model, "ic", None) is None:
            raise PostprocessingError(f"MF6 model {model_name} has no IC package")

        discretization = next(
            getattr(model, name)
            for name in ("dis", "disv", "disu")
            if getattr(model, name, None) is not None
        )
        idomain = discretization.idomain.array
        initial = np.asarray(model.ic.strt.array, dtype=np.float64).reshape(-1).copy()
        active = (
            np.ones(initial.size, dtype=bool)
            if idomain is None
            else np.asarray(idomain).reshape(-1) > 0
        )
        if not np.all(np.isfinite(initial[active])):
            raise PostprocessingError(
                f"initial heads contain non-finite values for {model_name}"
            )
        # Saved heads retain user-grid numbering, including inactive-cell
        # sentinels. Exclude those cells from statistics without renumbering.
        initial[~active] = np.nan

        head_path = _discover_output_path(
            model, "head_filerecord", config.mf6.workspace
        )
        if head_path is None or not head_path.is_file():
            fallback = config.mf6.workspace / f"{model_name.lower()}.hds"
            head_path = fallback if fallback.is_file() else head_path
        if head_path is None or not head_path.is_file():
            raise PostprocessingError(
                f"saved MF6 head file was not found for {model_name}"
            )

        reader = None
        errors: list[str] = []
        for reader_class in (HeadFile, HeadUFile):
            try:
                reader = reader_class(str(head_path), precision="double")
                times = np.asarray(reader.get_times(), dtype=np.float64)
                if times.size:
                    break
                reader.close()
                reader = None
            except Exception as exc:
                errors.append(f"{reader_class.__name__}: {exc}")
                if reader is not None:
                    reader.close()
                reader = None
        if reader is None:
            raise PostprocessingError(
                f"failed to read MF6 heads {head_path}: {'; '.join(errors)}"
            )

        resources.callback(reader.close)
        time_axis = np.concatenate(([0.0], times))
        result[model_name.upper()] = Mf6HeadSeries(
            model_name=model_name.upper(),
            times_days=time_axis,
            read_heads=partial(
                _read_head_record, reader, time_axis, initial, active, model_name
            ),
            initial_heads_m=initial,
        )
    return result


def _read_head_record(reader, times, initial, active, model_name, index):
    time_value = float(times[index])
    if index in (0, -len(times)):
        return initial
    array = np.asarray(reader.get_data(totim=time_value), dtype=np.float64).reshape(-1)
    if array.size != initial.size:
        raise PostprocessingError(
            f"MF6 head record size mismatch for {model_name} at t={time_value}: "
            f"expected={initial.size} actual={array.size}"
        )
    array[~active] = np.nan
    return array
