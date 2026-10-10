"""Prevent the types.py/logging.py collision that broke Python and editor startup."""

import os
import subprocess
import sys
from pathlib import Path

import vicmf6


def test_source_module_names_do_not_shadow_standard_library():
    root = Path(vicmf6.__file__).resolve().parent
    collisions = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.stem in sys.stdlib_module_names
        or (path.name == "__init__.py" and path.parent.name in sys.stdlib_module_names)
    ]
    assert not collisions, f"Standard library names in source directories: {collisions}"


def test_cli_sets_thread_defaults_before_scientific_imports():
    script = """
import os, sys
from vicmf6.cli import main
assert 'numpy' not in sys.modules
main(['version'])
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    expected = sys.argv[1] if name == 'OMP_NUM_THREADS' else '1'
    assert os.environ[name] == expected, (name, os.environ[name])
assert os.environ['OMP_DYNAMIC'] == 'FALSE'
"""
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_DYNAMIC",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(Path(vicmf6.__file__).resolve().parents[1])
    for threads in ("1", "3"):
        if threads == "3":
            environment["OMP_NUM_THREADS"] = threads
        subprocess.run(
            [sys.executable, "-B", "-c", script, threads],
            env=environment,
            check=True,
            capture_output=True,
            timeout=30,
        )
