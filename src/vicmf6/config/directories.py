"""Reserve coupler-owned outputs before any rank opens logs or native models.

The run root may contain prepared inputs. Its output subdirectories must be new;
exclusive creation also prevents two launches from claiming the same outputs."""

from __future__ import annotations

from ..errors import ConfigurationError
from .records import ApplicationConfig


def create_fresh_run_output_directories(config: ApplicationConfig) -> None:
    """reserve new output directories before any rank opens logs or models."""

    directories = (
        config.coupling.diagnostics_directory,
        config.vic.outputs_directory,
        config.vic.exchange_directory,
    )
    for directory in directories:
        if directory.exists():
            raise ConfigurationError(
                f"run output directory already exists: {directory}; choose a new run directory"
            )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=False)
