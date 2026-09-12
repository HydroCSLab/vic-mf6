"""launch one VIC Image Driver window as an MPI child of controller rank zero."""

from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

from .errors import VicRuntimeError


def spawn_vic(
    *,
    executable: Path,
    working_directory: Path,
    global_parameter_file: Path,
    mpi_processes: int,
    omp_threads: int,
    timeout_seconds: int,
    preload_library: Path | None,
    environment: dict[str, str],
    logger: object,
) -> None:
    """spawn VIC and wait for the child communicator to disconnect cleanly."""

    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise VicRuntimeError(
            "mpi4py is required to spawn the VIC child process; install vicmf6[runtime]"
        ) from exc

    if mpi_processes < 1 or omp_threads < 1 or timeout_seconds < 1:
        raise VicRuntimeError(
            "VIC process, thread, and timeout values must all be positive"
        )
    for path, label in (
        (executable, "VIC executable"),
        (global_parameter_file, "VIC global parameter file"),
    ):
        if not path.is_file():
            raise VicRuntimeError(f"{label} was not found: {path}")
    if not working_directory.is_dir():
        raise VicRuntimeError(
            f"VIC working directory was not found: {working_directory}"
        )
    if preload_library is not None and not preload_library.is_file():
        raise VicRuntimeError(f"VIC preload library was not found: {preload_library}")

    child_environment = dict(environment)
    child_environment.update(
        {
            "OMP_NUM_THREADS": str(omp_threads),
            "OMP_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(omp_threads),
            "NUMEXPR_NUM_THREADS": str(omp_threads),
        }
    )
    if preload_library is not None:
        child_environment["LD_PRELOAD"] = str(preload_library)

    # rank zero also uses a conservative thread environment. this avoids a
    # controller-side blas implementation consuming cores while child VIC and
    # persistent groundwater workers are active on the same allocation.
    for key in (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = child_environment[key]

    # do not encode the VIC environment in MPI_Info. MPI info values have a
    # small implementation-defined maximum length, while the coupling paths
    # alone can exceed it. launch through `env` instead: it sets the exact
    # child-only variables and then execs VIC before VIC enters MPI_Init.
    env_executable = shutil.which("env")
    if env_executable is None:
        raise VicRuntimeError("the 'env' executable is required to launch VIC")
    spawn_args = _environment_spawn_args(
        executable=executable,
        global_parameter_file=global_parameter_file,
        environment=child_environment,
    )

    info = MPI.Info.Create()
    try:
        info.Set("wdir", str(working_directory))
        _log(
            logger,
            "info",
            f"vic spawn ranks={mpi_processes} omp_threads={omp_threads} global={str(global_parameter_file.resolve())}",
        )
        _spawn_with_timeout(
            MPI=MPI,
            command=Path(env_executable),
            args=spawn_args,
            maxprocs=mpi_processes,
            info=info,
            timeout_seconds=timeout_seconds,
        )
    finally:
        info.Free()


def _environment_spawn_args(
    *,
    executable: Path,
    global_parameter_file: Path,
    environment: dict[str, str],
) -> list[str]:
    """build deterministic argv for `env ... vic -g global`."""

    assignments = [f"{key}={environment[key]}" for key in sorted(environment)]
    return assignments + [str(executable), "-g", str(global_parameter_file.resolve())]


def _spawn_with_timeout(
    *,
    MPI: object,
    command: Path,
    args: list[str],
    maxprocs: int,
    info: object,
    timeout_seconds: int,
) -> None:
    def alarm_handler(signum: int, frame: object) -> None:
        del signum, frame
        raise TimeoutError(
            "timeout while waiting for the VIC child ranks to disconnect"
        )

    previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout_seconds)
    try:
        intercommunicator = MPI.COMM_SELF.Spawn(
            str(command),
            args=args,
            maxprocs=maxprocs,
            info=info,
        )
        intercommunicator.Disconnect()
    except TimeoutError as exc:
        raise VicRuntimeError(str(exc)) from exc
    except Exception as exc:
        raise VicRuntimeError(f"VIC MPI spawn failed: {exc}") from exc
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _log(logger: object, level: str, message: str) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message)
