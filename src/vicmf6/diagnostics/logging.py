"""Create per-rank logs and controller-only progress output.

Messages from different ranks should not overwrite each other or obscure the
scientific sequence. The caller reserves fresh directories before opening logs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


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
