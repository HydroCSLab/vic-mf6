from __future__ import annotations

from pathlib import Path

from vicmf6.mpi_spawn import _environment_spawn_args


def test_environment_spawn_args_are_explicit_and_deterministic() -> None:
    args = _environment_spawn_args(
        executable=Path("/opt/vic/vic_image.exe"),
        global_parameter_file=Path("/tmp/window-0000.global.txt"),
        environment={
            "VIC_GW_HEAD_FILE": "/tmp/head.txt",
            "OMP_NUM_THREADS": "1",
            "LD_PRELOAD": "/opt/vicmf6/libdisconnect.so",
        },
    )

    assert args == [
        "LD_PRELOAD=/opt/vicmf6/libdisconnect.so",
        "OMP_NUM_THREADS=1",
        "VIC_GW_HEAD_FILE=/tmp/head.txt",
        "/opt/vic/vic_image.exe",
        "-g",
        "/tmp/window-0000.global.txt",
    ]
