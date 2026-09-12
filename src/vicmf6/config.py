"""load the small public yaml and resolve model-owned metadata before runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError
from .model_inputs import (
    Mf6SimulationMetadata,
    VicGlobalMetadata,
    parse_mf6_simulation,
    parse_vic_global,
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
                f"no resolved solution group for MF6 model {key}"
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


def load_config(path: str | Path, *, check_paths: bool = True) -> ApplicationConfig:
    """read the public yaml and derive everything already owned by VIC or MF6 inputs."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigurationError(f"configuration file was not found: {source_path}")

    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"failed to parse yaml: {source_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("top-level configuration must be a mapping")

    config_dir = source_path.parent
    run_raw = raw.get("run", {}) or {}
    if not isinstance(run_raw, dict):
        raise ConfigurationError("run must be a mapping")
    run_directory = _config_path(config_dir, run_raw.get("directory", "run"))

    mf6_raw = _require_mapping(raw, "mf6")
    vic_raw = _require_mapping(raw, "vic")
    coupling_raw = _require_mapping(raw, "coupling")
    diagnostics_raw = raw.get("diagnostics", {}) or {}
    if not isinstance(diagnostics_raw, dict):
        raise ConfigurationError("diagnostics must be a mapping")

    # the yaml points at the two model entry files. their own inputs remain the
    # authoritative source for calendar, grid-file paths, output declarations,
    # model names, API package names, solution groups, and TDIS metadata.
    vic_global_file = _config_path(config_dir, _required(vic_raw, "global_file", "vic"))
    mf6_namefile = _config_path(config_dir, _required(mf6_raw, "namefile", "mf6"))

    vic_source = parse_vic_global(vic_global_file)
    mf6_source = parse_mf6_simulation(mf6_namefile)

    mf6 = Mf6Config(
        workspace=mf6_source.workspace,
        library=_config_path(config_dir, _required(mf6_raw, "library", "mf6")),
        simulation_namefile=mf6_source.namefile.name,
        start_time=vic_source.start_time,
        api_packages={model.name: model.api_package for model in mf6_source.models},
        solution_ids={model.name: model.solution_id for model in mf6_source.models},
        total_time_days=mf6_source.total_time_days,
        time_step_boundaries_days=mf6_source.time_step_boundaries_days(),
        max_solve_iterations=_positive_int(
            mf6_raw.get("max_solve_iterations", 100), "mf6.max_solve_iterations"
        ),
    )

    coupling = CouplingConfig(
        exchange_table=_config_path(
            config_dir, _required(coupling_raw, "exchange_table", "coupling")
        ),
        start_time=vic_source.start_time,
        end_time=vic_source.end_time,
        interval_days=_positive_float(
            _required(coupling_raw, "interval_days", "coupling"),
            "coupling.interval_days",
        ),
        diagnostics_directory=run_directory / "diagnostics",
        scheme=_text(coupling_raw.get("scheme", "explicit"), "coupling.scheme").lower(),
        require_full_vic_coverage=_bool(
            coupling_raw.get("require_full_vic_coverage", True),
            "coupling.require_full_vic_coverage",
        ),
        coverage_relative_tolerance=_nonnegative_float(
            coupling_raw.get("coverage_relative_tolerance", 1.0e-10),
            "coupling.coverage_relative_tolerance",
        ),
        conservation_absolute_tolerance_m3=_nonnegative_float(
            coupling_raw.get("conservation_absolute_tolerance_m3", 1.0e-6),
            "coupling.conservation_absolute_tolerance_m3",
        ),
        conservation_relative_tolerance=_nonnegative_float(
            coupling_raw.get("conservation_relative_tolerance", 1.0e-12),
            "coupling.conservation_relative_tolerance",
        ),
        api_absolute_tolerance_m3_per_day=_nonnegative_float(
            coupling_raw.get("api_absolute_tolerance_m3_per_day", 1.0e-6),
            "coupling.api_absolute_tolerance_m3_per_day",
        ),
    )

    vic = VicConfig(
        working_directory=vic_source.working_directory,
        executable=_config_path(config_dir, _required(vic_raw, "executable", "vic")),
        global_file=vic_source.global_file,
        outputs_directory=run_directory / "vic" / "outputs",
        exchange_directory=run_directory / "vic" / "exchange",
        parameters_file=vic_source.parameters_file,
        domain_file=vic_source.domain_file,
        forcing_prefixes=vic_source.forcing_prefixes,
        exchange_variable="OUT_GW_EXCHANGE",
        exchange_output_prefix=vic_source.exchange_output_prefix,
        mpi_processes=_positive_int(
            vic_raw.get("mpi_processes", 1), "vic.mpi_processes"
        ),
        omp_threads=_positive_int(vic_raw.get("omp_threads", 1), "vic.omp_threads"),
        spawn_timeout_seconds=_positive_int(
            vic_raw.get("spawn_timeout_seconds", 3600), "vic.spawn_timeout_seconds"
        ),
        preload_library=_optional_path(config_dir, vic_raw.get("preload_library")),
        exchange_length_m=_positive_float(
            _required(coupling_raw, "exchange_length_m", "coupling"),
            "coupling.exchange_length_m",
        ),
        exchange_conductivity_scale=_positive_float(
            _required(coupling_raw, "exchange_conductivity_scale", "coupling"),
            "coupling.exchange_conductivity_scale",
        ),
        head_transform=_text(
            _required(coupling_raw, "head_transform", "coupling"),
            "coupling.head_transform",
        ).lower(),
        initial_state=vic_source.initial_state,
        environment=_environment(vic_raw.get("environment", {})),
    )

    diagnostics = DiagnosticsConfig(
        verbosity=_text(
            diagnostics_raw.get("verbosity", "info"), "diagnostics.verbosity"
        ).lower(),
        write_rank_logs=_bool(
            diagnostics_raw.get("write_rank_logs", True), "diagnostics.write_rank_logs"
        ),
    )

    config = ApplicationConfig(
        source_path=source_path,
        run_directory=run_directory,
        mf6=mf6,
        vic=vic,
        coupling=coupling,
        diagnostics=diagnostics,
        vic_source=vic_source,
        mf6_source=mf6_source,
    )
    _validate_cross_section_contract(config)
    if check_paths:
        _validate_paths(config)
    return config


def _validate_cross_section_contract(config: ApplicationConfig) -> None:
    if config.coupling.scheme != "explicit":
        raise ConfigurationError(
            "coupling.scheme must be 'explicit' in version 0.1; iterative coupling is not exposed until general rollback/replay is verified"
        )
    if config.coupling.end_time <= config.coupling.start_time:
        raise ConfigurationError(
            "the VIC global file describes an empty simulation period"
        )
    if config.vic.head_transform not in {
        "identity",
        "pressure_head_from_interface_elevation",
    }:
        raise ConfigurationError(
            "coupling.head_transform must be 'identity' or 'pressure_head_from_interface_elevation'"
        )
    if config.diagnostics.verbosity not in {"debug", "info", "warning", "error"}:
        raise ConfigurationError(
            "diagnostics.verbosity must be debug, info, warning, or error"
        )

    vic_duration_days = config.vic_source.duration_days
    mf6_duration_days = config.mf6_source.total_time_days
    duration_error = abs(vic_duration_days - mf6_duration_days)
    if duration_error > 1.0e-10 * max(vic_duration_days, mf6_duration_days, 1.0):
        raise ConfigurationError(
            "VIC and MF6 simulation durations do not match: "
            f"vic={vic_duration_days:.17g} days mf6={mf6_duration_days:.17g} days"
        )

    reserved_vic_environment = {
        "VIC_GW_FORMULATION",
        "VIC_GW_HEAD_FILE",
        "VIC_GW_EXCHANGE_LENGTH_M",
        "VIC_GW_TEST_KA_SCALE",
        "VIC_GW_MAX_DRAIN_FRACTION",
        "VIC_GW_REFERENCE_DEPTH",
        "VIC_ALLOW_PARTIAL_DAY",
    }
    conflicts = sorted(reserved_vic_environment.intersection(config.vic.environment))
    if conflicts:
        raise ConfigurationError(
            "vic.environment must not override coupler-owned groundwater controls: "
            + ", ".join(conflicts)
        )


def _validate_paths(config: ApplicationConfig) -> None:
    required_files = {
        "mf6.library": config.mf6.library,
        "mf6.namefile": config.mf6_source.namefile,
        "vic.executable": config.vic.executable,
        "vic.global_file": config.vic.global_file,
        "vic.parameters": config.vic.parameters_file,
        "vic.domain": config.vic.domain_file,
        "coupling.exchange_table": config.coupling.exchange_table,
    }
    for field_name, path in required_files.items():
        if not path.is_file():
            raise ConfigurationError(f"{field_name} was not found: {path}")

    if config.vic.initial_state is not None and not config.vic.initial_state.is_file():
        raise ConfigurationError(
            f"VIC INIT_STATE was not found: {config.vic.initial_state}"
        )
    if (
        config.vic.preload_library is not None
        and not config.vic.preload_library.is_file()
    ):
        raise ConfigurationError(
            f"vic.preload_library was not found: {config.vic.preload_library}"
        )


def create_run_directories(config: ApplicationConfig) -> None:
    """create only directories that are owned by the coupled run."""

    config.run_directory.mkdir(parents=True, exist_ok=True)
    config.vic.outputs_directory.mkdir(parents=True, exist_ok=True)
    config.vic.exchange_directory.mkdir(parents=True, exist_ok=True)
    config.coupling.diagnostics_directory.mkdir(parents=True, exist_ok=True)


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


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(
            f"missing required configuration field: {section}.{key}"
        )
    return mapping[key]


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _expand(raw: Any) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(raw)))
    if "$" in expanded:
        raise ConfigurationError(
            f"path contains an unresolved environment variable: {raw!r}"
        )
    return expanded


def _config_path(base: Path, raw: Any) -> Path:
    candidate = Path(_expand(raw))
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


def _optional_path(base: Path, raw: Any) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    return _config_path(base, raw)


def _text(raw: Any, field_name: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ConfigurationError(f"{field_name} must not be empty")
    return value


def _positive_int(raw: Any, field_name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be an integer") from exc
    if value < 1:
        raise ConfigurationError(f"{field_name} must be at least 1")
    return value


def _positive_float(raw: Any, field_name: str) -> float:
    value = _float(raw, field_name)
    if value <= 0.0:
        raise ConfigurationError(f"{field_name} must be greater than zero")
    return value


def _nonnegative_float(raw: Any, field_name: str) -> float:
    value = _float(raw, field_name)
    if value < 0.0:
        raise ConfigurationError(f"{field_name} must be nonnegative")
    return value


def _float(raw: Any, field_name: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be numeric") from exc


def _bool(raw: Any, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"true", "yes", "1", "on"}:
            return True
        if value in {"false", "no", "0", "off"}:
            return False
    raise ConfigurationError(f"{field_name} must be true or false")


def _environment(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError("vic.environment must be a mapping")
    return {str(key): str(value) for key, value in raw.items()}
