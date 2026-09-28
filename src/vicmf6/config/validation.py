"""Validate configuration at the boundary, before MPI initializes models.

Numerical tolerances must be finite: comparing an error against NaN would
silently disable a conservation check. Path expansion is explicit and relative
to the configuration file, so launching from another directory is reproducible."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from .records import ApplicationConfig


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
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(value):
        raise ConfigurationError(f"{field_name} must be finite")
    return value


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
