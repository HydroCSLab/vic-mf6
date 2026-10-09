"""Controller-only CSV, manifest, and summary output.

Workers contribute numerical quantities through MPI; they never append to the
shared CSV. Keeping one writer makes output order deterministic across ranks."""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import ApplicationConfig, config_as_dict
from .records import WindowDiagnostics


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
    from .. import __version__

    return __version__
