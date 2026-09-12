"""cheap static checks that run before MPI/XMI/VIC consume expensive resources."""

from __future__ import annotations

import math

from .config import ApplicationConfig
from .errors import ConfigurationError
from .exchange import ExchangeTable
from .schedule import build_windows, vic_record_count


def run_preflight(config: ApplicationConfig) -> dict[str, object]:
    """validate discovered model metadata, geometry, and coupling-window compatibility."""

    exchange = ExchangeTable.from_csv(
        config.coupling.exchange_table,
        require_full_vic_coverage=config.coupling.require_full_vic_coverage,
        coverage_relative_tolerance=config.coupling.coverage_relative_tolerance,
    )

    model_names = [model.name for model in config.mf6_source.models]
    if {name.casefold() for name in exchange.model_names} != {
        name.casefold() for name in model_names
    }:
        raise ConfigurationError(
            f"exchange table models {exchange.model_names} do not match MF6 models {model_names}"
        )

    windows = build_windows(
        config.coupling.start_time,
        config.coupling.end_time,
        config.coupling.interval_days,
    )
    for window in windows:
        vic_record_count(config.vic_source.model_steps_per_day, window.duration_days)

    _validate_mf6_coupling_boundaries(config, windows)

    if config.vic.head_transform == "pressure_head_from_interface_elevation":
        _ = exchange.vic_interface_elevations_m()

    return {
        "vic": {
            "global_file": str(config.vic_source.global_file),
            "domain_file": str(config.vic_source.domain_file),
            "parameters_file": str(config.vic_source.parameters_file),
            "forcing_streams": len(config.vic_source.forcing_prefixes),
            "model_steps_per_day": config.vic_source.model_steps_per_day,
            "nrecs": config.vic_source.nrecs,
            "start_time": config.vic_source.start_time.isoformat(),
            "end_time": config.vic_source.end_time.isoformat(),
            "exchange_output_prefix": config.vic_source.exchange_output_prefix,
            "exchange_variable": config.vic.exchange_variable,
        },
        "mf6": {
            "namefile": str(config.mf6_source.namefile),
            "tdis_file": str(config.mf6_source.tdis_file),
            "time_units": config.mf6_source.time_units,
            "total_time_days": config.mf6_source.total_time_days,
            "models": [
                {
                    "name": model.name,
                    "namefile": str(model.namefile),
                    "api_package": model.api_package,
                    "solution_id": model.solution_id,
                }
                for model in config.mf6_source.models
            ],
        },
        "coupling": {
            "windows": len(windows),
            "interval_days": config.coupling.interval_days,
            "vic_cells": exchange.vic_cell_count,
            "overlap_rows": exchange.overlap_count,
            "total_overlap_area_m2": exchange.total_overlap_area_m2,
        },
        "mpi": {
            "controller_ranks": 1,
            "mf6_worker_ranks": len(model_names),
            "world_ranks_required": len(model_names) + 1,
            "vic_child_ranks": config.vic.mpi_processes,
        },
    }


def _validate_mf6_coupling_boundaries(
    config: ApplicationConfig, windows: object
) -> None:
    boundaries = config.mf6_source.time_step_boundaries_days()
    tolerance = 1.0e-10
    for window in windows:
        requested = (window.end - config.coupling.start_time).total_seconds() / 86400.0
        if not any(
            math.isclose(requested, value, rel_tol=0.0, abs_tol=tolerance)
            for value in boundaries
        ):
            raise ConfigurationError(
                "an explicit coupling boundary does not coincide with an MF6 time-step boundary: "
                f"window={window.index + 1} boundary={requested:.17g} days"
            )
