"""rank-aware logging, run provenance, and compact scientific diagnostics."""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ApplicationConfig, config_as_dict
from .exchange import SignedVolume
from .schedule import CouplingWindow


@dataclass(frozen=True, slots=True)
class WindowDiagnostics:
    """one row in the coupled numerical and performance record."""

    step: int
    start: str
    end: str
    vic_positive_m3: float
    vic_negative_m3: float
    vic_net_m3: float
    mapped_positive_m3: float
    mapped_negative_m3: float
    mapped_net_m3: float
    boundary_positive_m3: float
    boundary_negative_m3: float
    boundary_net_m3: float
    applied_positive_m3: float
    applied_negative_m3: float
    applied_net_m3: float
    positive_mapping_error_m3: float
    negative_mapping_error_m3: float
    net_mapping_error_m3: float
    aggregation_net_error_m3: float
    aggregation_positive_cancellation_m3: float
    aggregation_negative_cancellation_m3: float
    positive_api_error_m3: float
    negative_api_error_m3: float
    net_api_error_m3: float
    head_min_m: float
    head_max_m: float
    head_mean_m: float
    maximum_api_rate_error_m3_per_day: float
    maximum_vic_water_error_mm: float | None
    lateral_domain_net_m3: float | None
    lateral_gross_pair_m3: float | None
    lateral_maximum_pair_antisymmetry_m3: float | None
    nonlinear_iterations_max: int
    vic_prepare_seconds: float
    vic_seconds: float
    mf6_seconds: float
    mapping_seconds: float
    mpi_seconds: float
    io_seconds: float
    window_seconds: float


class DiagnosticsWriter:
    """write deterministic CSV/JSON artifacts owned by controller rank zero."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.window_path = directory / "coupling_windows.csv"
        self.summary_path = directory / "run_summary.json"
        self.manifest_path = directory / "run_manifest.json"
        if self.window_path.exists():
            self.window_path.unlink()
        self._rows: list[WindowDiagnostics] = []

    def write_manifest(
        self,
        config: ApplicationConfig,
        *,
        world_size: int,
        mf6_models: list[str],
    ) -> None:
        manifest = {
            "vicmf6_version": _package_version(),
            "config": config_as_dict(config),
            "world_size": world_size,
            "mf6_models": mf6_models,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "cwd": os.getcwd(),
            "selected_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OMPI_MCA_rmaps_base_oversubscribe",
                    "PMIX_MCA_gds",
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NODELIST",
                )
                if os.environ.get(key) is not None
            },
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def append(self, row: WindowDiagnostics) -> None:
        write_header = not self.window_path.exists()
        with self.window_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(row)))
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(row))
        self._rows.append(row)

    def write_summary(self, *, total_wall_seconds: float) -> None:
        cumulative = {
            "vic": _sum_signed(self._rows, "vic"),
            "mapped": _sum_signed(self._rows, "mapped"),
            "boundary": _sum_signed(self._rows, "boundary"),
            "applied": _sum_signed(self._rows, "applied"),
        }
        cumulative_errors = {
            "mapping": _signed_difference(cumulative["mapped"], cumulative["vic"]),
            "aggregation": _signed_difference(
                cumulative["boundary"], cumulative["mapped"]
            ),
            "api": _signed_difference(cumulative["applied"], cumulative["boundary"]),
            "cross_model": _signed_difference(cumulative["applied"], cumulative["vic"]),
        }
        summary: dict[str, Any] = {
            "windows": len(self._rows),
            "total_wall_seconds": total_wall_seconds,
            "cumulative": cumulative,
            "cumulative_errors_m3": cumulative_errors,
            "relative_cross_model_net_error": _relative_net_error(
                cumulative["vic"]["net_m3"], cumulative["applied"]["net_m3"]
            ),
            "timing_seconds": {
                "vic": sum(row.vic_seconds for row in self._rows),
                "mf6": sum(row.mf6_seconds for row in self._rows),
                "mapping": sum(row.mapping_seconds for row in self._rows),
                "mpi": sum(row.mpi_seconds for row in self._rows),
                "io": sum(row.io_seconds for row in self._rows),
            },
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def build_logger(
    *,
    rank: int,
    diagnostics_directory: Path,
    level_name: str,
    write_rank_logs: bool,
) -> logging.Logger:
    """create one rank logger while keeping user-facing progress on rank zero."""

    logger = logging.getLogger(f"vicmf6.rank{rank}")
    logger.handlers.clear()
    logger.propagate = False
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt=f"%(asctime)s rank={rank} %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if rank == 0:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(level)
        logger.addHandler(console)

    if write_rank_logs:
        diagnostics_directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            diagnostics_directory / f"rank-{rank:04d}.log", mode="w", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    return logger


def make_window_diagnostics(
    *,
    window: CouplingWindow,
    vic: SignedVolume,
    mapped: SignedVolume,
    boundary_target: SignedVolume,
    applied: SignedVolume,
    head_min_m: float,
    head_max_m: float,
    head_mean_m: float,
    maximum_api_rate_error_m3_per_day: float,
    maximum_vic_water_error_mm: float | None,
    lateral_domain_net_m3: float | None,
    lateral_gross_pair_m3: float | None,
    lateral_maximum_pair_antisymmetry_m3: float | None,
    nonlinear_iterations_max: int,
    vic_prepare_seconds: float,
    vic_seconds: float,
    mf6_seconds: float,
    mapping_seconds: float,
    mpi_seconds: float,
    io_seconds: float,
    window_seconds: float,
) -> WindowDiagnostics:
    return WindowDiagnostics(
        step=window.index + 1,
        start=window.start.isoformat(),
        end=window.end.isoformat(),
        vic_positive_m3=vic.positive_m3,
        vic_negative_m3=vic.negative_m3,
        vic_net_m3=vic.net_m3,
        mapped_positive_m3=mapped.positive_m3,
        mapped_negative_m3=mapped.negative_m3,
        mapped_net_m3=mapped.net_m3,
        boundary_positive_m3=boundary_target.positive_m3,
        boundary_negative_m3=boundary_target.negative_m3,
        boundary_net_m3=boundary_target.net_m3,
        applied_positive_m3=applied.positive_m3,
        applied_negative_m3=applied.negative_m3,
        applied_net_m3=applied.net_m3,
        positive_mapping_error_m3=mapped.positive_m3 - vic.positive_m3,
        negative_mapping_error_m3=mapped.negative_m3 - vic.negative_m3,
        net_mapping_error_m3=mapped.net_m3 - vic.net_m3,
        aggregation_net_error_m3=boundary_target.net_m3 - mapped.net_m3,
        aggregation_positive_cancellation_m3=(
            mapped.positive_m3 - boundary_target.positive_m3
        ),
        aggregation_negative_cancellation_m3=(
            boundary_target.negative_m3 - mapped.negative_m3
        ),
        positive_api_error_m3=applied.positive_m3 - boundary_target.positive_m3,
        negative_api_error_m3=applied.negative_m3 - boundary_target.negative_m3,
        net_api_error_m3=applied.net_m3 - boundary_target.net_m3,
        head_min_m=head_min_m,
        head_max_m=head_max_m,
        head_mean_m=head_mean_m,
        maximum_api_rate_error_m3_per_day=maximum_api_rate_error_m3_per_day,
        maximum_vic_water_error_mm=maximum_vic_water_error_mm,
        lateral_domain_net_m3=lateral_domain_net_m3,
        lateral_gross_pair_m3=lateral_gross_pair_m3,
        lateral_maximum_pair_antisymmetry_m3=lateral_maximum_pair_antisymmetry_m3,
        nonlinear_iterations_max=nonlinear_iterations_max,
        vic_prepare_seconds=vic_prepare_seconds,
        vic_seconds=vic_seconds,
        mf6_seconds=mf6_seconds,
        mapping_seconds=mapping_seconds,
        mpi_seconds=mpi_seconds,
        io_seconds=io_seconds,
        window_seconds=window_seconds,
    )


def _sum_signed(rows: list[WindowDiagnostics], prefix: str) -> dict[str, float]:
    return {
        "positive_m3": sum(getattr(row, f"{prefix}_positive_m3") for row in rows),
        "negative_m3": sum(getattr(row, f"{prefix}_negative_m3") for row in rows),
        "net_m3": sum(getattr(row, f"{prefix}_net_m3") for row in rows),
    }


def _signed_difference(
    actual: dict[str, float], expected: dict[str, float]
) -> dict[str, float]:
    return {
        "positive_m3": actual["positive_m3"] - expected["positive_m3"],
        "negative_m3": actual["negative_m3"] - expected["negative_m3"],
        "net_m3": actual["net_m3"] - expected["net_m3"],
    }


def _relative_net_error(expected_m3: float, actual_m3: float) -> float:
    scale = max(abs(expected_m3), abs(actual_m3))
    if scale == 0.0:
        return 0.0
    return (actual_m3 - expected_m3) / scale


def _package_version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:
        return "unknown"
