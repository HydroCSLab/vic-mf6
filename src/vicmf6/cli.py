"""command-line entry point for validation and coupled execution."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .diagnostics import build_logger
from .errors import VicMf6Error
from .preflight import run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "version":
        from . import __version__

        print(__version__)
        return 0

    try:
        config = load_config(
            args.config,
            check_paths=args.command not in {"inspect", "post"},
        )
        if getattr(args, "vic_processes", None) is not None:
            if args.vic_processes < 1:
                raise ValueError("--vic-processes must be at least 1")
            config = replace(
                config,
                vic=replace(config.vic, mpi_processes=args.vic_processes),
            )
        if getattr(args, "run_directory", None) is not None:
            run_directory = args.run_directory.expanduser().resolve()
            config = replace(
                config,
                run_directory=run_directory,
                vic=replace(
                    config.vic,
                    outputs_directory=run_directory / "vic" / "outputs",
                    exchange_directory=run_directory / "vic" / "exchange",
                ),
                coupling=replace(
                    config.coupling,
                    diagnostics_directory=run_directory / "diagnostics",
                ),
            )
    except (VicMf6Error, ValueError) as exc:
        print(f"vicmf6: {exc}", file=sys.stderr)
        return 2

    if args.command == "inspect":
        try:
            summary = run_preflight(config)
            from .inspection import format_inspection
        except (VicMf6Error, ValueError, OSError) as exc:
            print(f"vicmf6 inspect failed: {exc}", file=sys.stderr)
            return 2
        print(format_inspection(config, summary))
        return 0

    if args.command == "preflight":
        try:
            summary = run_preflight(config)
        except (VicMf6Error, ValueError, OSError) as exc:
            print(f"vicmf6 preflight failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "post":
        try:
            from .post import run_postprocessing

            formats = tuple(
                item.strip().lower() for item in args.formats.split(",") if item.strip()
            )
            if not formats:
                raise ValueError("--formats must contain at least one format")
            return run_postprocessing(
                config,
                action=args.post_command,
                output_directory=args.output_directory,
                figure_formats=formats,
                figure_dpi=args.dpi,
                strict_reference=args.strict_reference,
            )
        except (VicMf6Error, ValueError, OSError) as exc:
            print(f"vicmf6 post failed: {exc}", file=sys.stderr)
            return 2

    if args.command == "run":
        try:
            from mpi4py import MPI
        except ImportError:
            print(
                "vicmf6 run requires mpi4py; install vicmf6[runtime] and launch with mpirun",
                file=sys.stderr,
            )
            return 2
        rank = int(MPI.COMM_WORLD.Get_rank())
        logger = build_logger(
            rank=rank,
            diagnostics_directory=config.coupling.diagnostics_directory,
            level_name=config.diagnostics.verbosity,
            write_rank_logs=config.diagnostics.write_rank_logs,
        )
        try:
            if rank == 0:
                summary = run_preflight(config)
                logger.info(
                    "preflight passed "
                    f"mf6_models={len(summary['mf6']['models'])} vic_cells={summary['coupling']['vic_cells']} overlaps={summary['coupling']['overlap_rows']}"
                )
            MPI.COMM_WORLD.Barrier()
            from .driver import run_coupling

            return run_coupling(config, logger=logger)
        except Exception as exc:
            logger.exception(f"coupling failed: {exc}")
            # a single rank leaving a collective path can deadlock every peer.
            # abort the world so failure is immediate and carries rank context.
            MPI.COMM_WORLD.Abort(1)
            return 1

    raise AssertionError(f"unhandled command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vicmf6",
        description="MPI-parallel explicit two-way VIC-MODFLOW 6 coupler",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect = subparsers.add_parser(
        "inspect", help="show model metadata discovered from VIC and MF6 inputs"
    )
    inspect.add_argument("-c", "--config", required=True, type=Path)

    preflight = subparsers.add_parser(
        "preflight", help="validate configuration, mapping, and model inputs"
    )
    preflight.add_argument("-c", "--config", required=True, type=Path)

    run = subparsers.add_parser("run", help="run the coupled simulation")
    run.add_argument("-c", "--config", required=True, type=Path)
    run.add_argument(
        "--vic-processes",
        type=int,
        default=None,
        help="override vic.mpi_processes for this launch",
    )
    run.add_argument(
        "--run-directory",
        type=Path,
        default=None,
        help="place coupler-owned outputs under this directory for this launch",
    )

    post = subparsers.add_parser(
        "post", help="extract, validate, and visualize a completed coupled run"
    )
    post_subparsers = post.add_subparsers(dest="post_command", required=True)
    for name, help_text in (
        ("summarize", "write canonical tables, summaries, and acceptance outputs"),
        ("figures", "write canonical tables plus the complete figure suite"),
        ("accept", "evaluate numerical and optional archived-reference acceptance"),
        ("all", "write tables, acceptance outputs, report, and figures"),
    ):
        post_command = post_subparsers.add_parser(name, help=help_text)
        post_command.add_argument("-c", "--config", required=True, type=Path)
        post_command.add_argument(
            "--output-directory",
            type=Path,
            default=None,
            help="override the default RUN/postprocessing output directory",
        )
        post_command.add_argument(
            "--formats",
            default="png,pdf",
            help="comma-separated figure formats for figures/all (png,pdf,svg)",
        )
        post_command.add_argument(
            "--dpi",
            type=int,
            default=220,
            help="PNG resolution for figures/all",
        )
        post_command.add_argument(
            "--strict-reference",
            action="store_true",
            help="return failure if an available archived H8c reference comparison fails",
        )

    subparsers.add_parser("version", help="print the package version")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
