"""Resolve the public YAML into one validated run configuration.

VIC and MF6 input decks are authoritative for model metadata. YAML adds only
cross-model controls and executable locations, avoiding two competing calendars."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..errors import ConfigurationError
from ..model_inputs import (
    parse_mf6_simulation,
    parse_vic_global,
)
from .records import (
    ApplicationConfig,
    CouplingConfig,
    DiagnosticsConfig,
    Mf6Config,
    VicConfig,
)
from .validation import (
    _bool,
    _config_path,
    _environment,
    _nonnegative_float,
    _optional_path,
    _positive_float,
    _positive_int,
    _require_mapping,
    _required,
    _text,
    _validate_cross_section_contract,
    _validate_paths,
)


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
        surface_runoff_table=_optional_path(
            config_dir, coupling_raw.get("surface_runoff_table")
        ),
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
        runoff_variable=(
            "OUT_RUNOFF"
            if coupling.surface_runoff_table is not None
            else None
        ),
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
