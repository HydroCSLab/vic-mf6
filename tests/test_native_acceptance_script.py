import os
import subprocess
from pathlib import Path


def test_failed_native_step_preserves_exit_status_and_failure_artifact(tmp_path):
    prefix = tmp_path / "install"
    for name in ("venv/bin", "bin", "lib"):
        (prefix / name).mkdir(parents=True)
    for name in ("bin/vic_image.exe", "bin/mf6", "bin/mpirun"):
        executable = prefix / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    python = prefix / "venv/bin/python"
    # Fail the first model-preparation step; environment reporting still works.
    python.write_text('#!/bin/sh\nif [ "$1" = "-E" ]; then exit 17; fi\nexit 0\n')
    python.chmod(0o755)
    for name in ("libmf6.so", "libvic_parent_disconnect.so"):
        (prefix / "lib" / name).touch()

    output = tmp_path / "results"
    script = (
        Path(__file__).resolve().parents[1] / "bundle/scripts/run-native-acceptance.sh"
    )
    env = {**os.environ, "VICMF6_ACCEPTANCE_WORK_DIR": str(tmp_path / "work")}
    result = subprocess.run(
        ["bash", str(script), str(prefix), str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 17, result.stdout + result.stderr
    assert (output / "acceptance-status.txt").read_text() == "FAIL exit_code=17\n"
