"""Reconstruct VIC window totals from saved NetCDF output.

The same missing-data convention used at runtime must be retained here so
postprocessing cannot silently accept a window the coupler would reject."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ...config import ApplicationConfig
from ...errors import PostprocessingError, VicRuntimeError
from ...vic.outputs import read_vic_window_outputs


def load_vic_exchange_fields(
    config: ApplicationConfig,
    *,
    expected_windows: int,
) -> Iterator[dict[str, Any]]:
    """read the signed VIC exchange field accumulated in each coupling window."""

    prefix = config.vic.exchange_output_prefix
    if not prefix:
        raise PostprocessingError(
            "the VIC exchange output stream could not be resolved"
        )

    for index in range(expected_windows):
        tag = f"window-{index:04d}"
        directory = config.vic.outputs_directory / tag
        paths = sorted(directory.glob(f"{prefix}.*.nc"))
        if not paths:
            raise PostprocessingError(
                f"no VIC output matching {prefix}.*.nc was found for {tag}: {directory}"
            )

        try:
            total, water_error_max, _ = read_vic_window_outputs(
                directory, prefix=prefix, exchange_variable=config.vic.exchange_variable
            )
        except VicRuntimeError as exc:
            raise PostprocessingError(str(exc)) from exc
        yield {
            "index": index,
            "tag": tag,
            "field_mm": total,
            "water_error_max_mm": water_error_max,
            "files": tuple(paths),
        }
