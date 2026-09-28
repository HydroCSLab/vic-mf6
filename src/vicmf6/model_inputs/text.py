"""Small text-input rules shared by the VIC and MODFLOW metadata readers."""

from __future__ import annotations

from pathlib import Path

from ..errors import ConfigurationError


def _resolve_model_path(base: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


def _positive_int_text(raw: str, field_name: str) -> int:
    try:
        value = int(raw.split()[0])
    except (ValueError, IndexError) as exc:
        raise ConfigurationError(f"{field_name} must be an integer") from exc
    if value < 1:
        raise ConfigurationError(f"{field_name} must be at least 1")
    return value


def _nonnegative_int_text(raw: str, field_name: str) -> int:
    try:
        value = int(raw.split()[0])
    except (ValueError, IndexError) as exc:
        raise ConfigurationError(f"{field_name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{field_name} must be nonnegative")
    return value


def _strip_comment(raw_line: str) -> str:
    return raw_line.split("#", 1)[0].strip()
