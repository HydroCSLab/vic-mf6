"""Opt-in tests of actual MPI collectives and coordinated failure exit.

Enable with VICMF6_RUN_MPI_TESTS=1 in the supported Open MPI runtime. Each
scenario uses in-memory model adapters, so native binaries are not required.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(
        os.environ.get("VICMF6_RUN_MPI_TESTS") != "1",
        reason="set VICMF6_RUN_MPI_TESTS=1 to launch real three-rank MPI tests",
    ),
]


@pytest.mark.parametrize(
    "scenario", ["complete-topology", "partial-topology", "worker-failure"]
)
def test_explicit_scheme_on_two_groundwater_workers(scenario, tmp_path):
    launcher = shutil.which("mpiexec")
    assert launcher is not None, "Open MPI launcher is required for opted-in tests"
    script = Path(__file__).parent / "mpi" / "explicit_scenario.py"
    result = subprocess.run(
        [
            launcher,
            "--oversubscribe",
            "-n",
            "3",
            sys.executable,
            str(script),
            scenario,
            str(tmp_path),
        ],
        env={
            **os.environ,
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    if scenario == "worker-failure":
        assert result.returncode != 0, output
        assert "injected worker failure" in output
        assert not list(tmp_path.glob("finalized-*")), (
            "failure entered collective native cleanup"
        )
    else:
        assert result.returncode == 0, output
        assert f"PASS: {scenario}" in output
