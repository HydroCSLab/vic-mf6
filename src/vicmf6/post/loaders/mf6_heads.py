"""Read initial and saved groundwater heads in model-node order."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config import ApplicationConfig
from ...errors import PostprocessingError
from ..types import Mf6HeadSeries
from .mf6_metadata import _discover_output_path


def load_mf6_head_series(
    config: ApplicationConfig,
    simulation: Any,
) -> dict[str, Mf6HeadSeries]:
    """read initial heads and all saved MF6 head records for every GWF model."""

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

        initial = np.asarray(model.ic.strt.array, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(initial)):
            raise PostprocessingError(
                f"initial heads contain non-finite values for {model_name}"
            )

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
            except Exception as exc:
                errors.append(f"{reader_class.__name__}: {exc}")
                reader = None
        if reader is None:
            raise PostprocessingError(
                f"failed to read MF6 heads {head_path}: {'; '.join(errors)}"
            )

        saved: list[np.ndarray] = []
        for time_value in times:
            array = np.asarray(
                reader.get_data(totim=float(time_value)), dtype=np.float64
            ).reshape(-1)
            if array.size != initial.size:
                raise PostprocessingError(
                    f"MF6 head record size mismatch for {model_name} at t={time_value}: "
                    f"expected={initial.size} actual={array.size}"
                )
            saved.append(array)
        heads = np.vstack([initial, *saved]) if saved else initial.reshape(1, -1)
        time_axis = np.concatenate(([0.0], times))
        result[model_name.upper()] = Mf6HeadSeries(
            model_name=model_name.upper(),
            times_days=time_axis,
            heads_m=heads,
            initial_heads_m=initial,
        )
    return result
