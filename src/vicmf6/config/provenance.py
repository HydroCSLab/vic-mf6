"""Serialize resolved settings for the run manifest.

Only serializable metadata belongs here. Native pointers, communicators, and
model state never enter the provenance document."""

from __future__ import annotations

from typing import Any

from .types import ApplicationConfig


def config_as_dict(config: ApplicationConfig) -> dict[str, Any]:
    """return a json-safe resolved configuration for provenance."""

    return {
        "source_path": str(config.source_path),
        "run_directory": str(config.run_directory),
        "mf6": {
            "namefile": str(config.mf6_source.namefile),
            "workspace": str(config.mf6.workspace),
            "library": str(config.mf6.library),
            "start_time": config.mf6.start_time.isoformat(),
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
            "max_solve_iterations": config.mf6.max_solve_iterations,
        },
        "vic": {
            "working_directory": str(config.vic.working_directory),
            "executable": str(config.vic.executable),
            "global_file": str(config.vic.global_file),
            "domain_file": str(config.vic.domain_file),
            "parameters_file": str(config.vic.parameters_file),
            "forcing_prefixes": [str(path) for path in config.vic.forcing_prefixes],
            "model_steps_per_day": config.vic_source.model_steps_per_day,
            "nrecs": config.vic_source.nrecs,
            "start_time": config.vic_source.start_time.isoformat(),
            "end_time": config.vic_source.end_time.isoformat(),
            "outputs_directory": str(config.vic.outputs_directory),
            "exchange_directory": str(config.vic.exchange_directory),
            "exchange_variable": config.vic.exchange_variable,
            "exchange_output_prefix": config.vic.exchange_output_prefix,
            "mpi_processes": config.vic.mpi_processes,
            "omp_threads": config.vic.omp_threads,
            "spawn_timeout_seconds": config.vic.spawn_timeout_seconds,
            "preload_library": str(config.vic.preload_library)
            if config.vic.preload_library
            else None,
            "initial_state": str(config.vic.initial_state)
            if config.vic.initial_state
            else None,
            "environment_keys": sorted(config.vic.environment),
        },
        "coupling": {
            "exchange_table": str(config.coupling.exchange_table),
            "start_time": config.coupling.start_time.isoformat(),
            "end_time": config.coupling.end_time.isoformat(),
            "interval_days": config.coupling.interval_days,
            "scheme": config.coupling.scheme,
            "exchange_length_m": config.vic.exchange_length_m,
            "exchange_conductivity_scale": config.vic.exchange_conductivity_scale,
            "head_transform": config.vic.head_transform,
            "diagnostics_directory": str(config.coupling.diagnostics_directory),
            "require_full_vic_coverage": config.coupling.require_full_vic_coverage,
            "coverage_relative_tolerance": config.coupling.coverage_relative_tolerance,
            "conservation_absolute_tolerance_m3": config.coupling.conservation_absolute_tolerance_m3,
            "conservation_relative_tolerance": config.coupling.conservation_relative_tolerance,
            "api_absolute_tolerance_m3_per_day": config.coupling.api_absolute_tolerance_m3_per_day,
        },
        "diagnostics": {
            "verbosity": config.diagnostics.verbosity,
            "write_rank_logs": config.diagnostics.write_rank_logs,
        },
    }
