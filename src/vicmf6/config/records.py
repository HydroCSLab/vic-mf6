"""Resolved configuration shared by all components.

These records contain paths, units, and numerical settings, not active model
resources. Freezing the records prevents a worker from changing the run contract
after validation. Model-owned dates and identifiers are resolved by the loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..errors import ConfigurationError
from ..model_inputs import (
    Mf6SimulationMetadata,
    VicGlobalMetadata,
)


@dataclass(frozen=True, slots=True)
class Mf6Config:
    """fully resolved persistent MODFLOW 6 runtime settings."""

    workspace: Path
    library: Path
    simulation_namefile: str
    start_time: datetime
    api_packages: dict[str, str]
    solution_ids: dict[str, int]
    total_time_days: float
    time_step_boundaries_days: tuple[float, ...]
    max_solve_iterations: int = 100

    def api_package_for(self, model_name: str) -> str:
        key = str(model_name).upper()
        try:
            return self.api_packages[key]
        except KeyError as exc:
            raise ConfigurationError(
                f"no resolved API6 package for MF6 model {key}"
            ) from exc

    def solution_id_for(self, model_name: str) -> int:
        key = str(model_name).upper()
        try:
            return self.solution_ids[key]
        except KeyError as exc:
            raise ConfigurationError(
                f"no resolved solver ID for MF6 model {key}"
            ) from exc


@dataclass(frozen=True, slots=True)
class VicConfig:
    """fully resolved VIC Image Driver runtime settings."""

    working_directory: Path
    executable: Path
    global_file: Path
    outputs_directory: Path
    exchange_directory: Path
    parameters_file: Path
    domain_file: Path
    forcing_prefixes: tuple[Path, ...]
    exchange_length_m: float
    exchange_conductivity_scale: float
    head_transform: str
    exchange_variable: str = "OUT_GW_EXCHANGE"
    runoff_variable: str | None = None
    exchange_output_prefix: str | None = None
    mpi_processes: int = 1
    omp_threads: int = 1
    spawn_timeout_seconds: int = 3600
    preload_library: Path | None = None
    initial_state: Path | None = None
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CouplingConfig:
    """cross-model time, mapping, and numerical acceptance settings."""

    exchange_table: Path
    start_time: datetime
    end_time: datetime
    interval_days: float
    diagnostics_directory: Path
    surface_runoff_table: Path | None = None
    scheme: str = "explicit"
    require_full_vic_coverage: bool = True
    coverage_relative_tolerance: float = 1.0e-10
    conservation_absolute_tolerance_m3: float = 1.0e-6
    conservation_relative_tolerance: float = 1.0e-12
    api_absolute_tolerance_m3_per_day: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """runtime reporting controls."""

    verbosity: str = "info"
    write_rank_logs: bool = True


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """fully resolved configuration used by preflight and the production runtime."""

    source_path: Path
    run_directory: Path
    mf6: Mf6Config
    vic: VicConfig
    coupling: CouplingConfig
    diagnostics: DiagnosticsConfig
    vic_source: VicGlobalMetadata
    mf6_source: Mf6SimulationMetadata
