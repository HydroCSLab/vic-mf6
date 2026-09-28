"""Command-line entry point for offline exchange-table generation."""

from __future__ import annotations

import argparse
import sys

from .intersections import build_exchange_table
from .types import ExchangeBuildError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build and validate an immutable VIC-MF6 polygon-overlap table"
    )
    parser.add_argument(
        "--vic-global", required=True, help="VIC Image Driver global file"
    )
    parser.add_argument("--mf6-sim", required=True, help="MODFLOW 6 mfsim.nam")
    parser.add_argument("--output", required=True, help="exchange-table CSV to create")
    parser.add_argument(
        "--mf6-crs",
        help="override/define the MF6 horizontal CRS (for example EPSG:5070)",
    )
    parser.add_argument(
        "--interface-elevation-m",
        type=float,
        help="constant VIC soil-base elevation in the shared vertical datum",
    )
    parser.add_argument(
        "--allow-partial-vic-coverage",
        action="store_true",
        help="allow active VIC cells to be only partly covered by MF6",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=5.0e-4,
        help="absolute tolerance on geometric coverage fraction (default: 5e-4)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_exchange_table(
            vic_global=args.vic_global,
            mf6_sim=args.mf6_sim,
            output=args.output,
            mf6_crs=args.mf6_crs,
            interface_elevation_m=args.interface_elevation_m,
            require_full_vic_coverage=not args.allow_partial_vic_coverage,
            coverage_tolerance=args.coverage_tolerance,
            force=args.force,
        )
    except ExchangeBuildError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0
