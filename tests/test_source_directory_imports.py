"""Editors and Python must start safely from any package directory.

An empty PYTHONPATH entry adds the current directory even during site startup.
A local types.py can then replace the standard library module before the tool
gets far enough to import the framework or display its own error message.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import vicmf6

PACKAGE_ROOT = Path(vicmf6.__file__).resolve().parent
PACKAGE_DIRECTORIES = sorted({path.parent for path in PACKAGE_ROOT.rglob("*.py")})


def test_source_module_names_do_not_shadow_standard_library():
    collisions = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.stem in sys.stdlib_module_names
        or (path.name == "__init__.py" and path.parent.name in sys.stdlib_module_names)
    ]
    assert not collisions, f"Standard library names in source directories: {collisions}"


@pytest.mark.parametrize(
    "directory",
    PACKAGE_DIRECTORIES,
    ids=lambda path: str(path.relative_to(PACKAGE_ROOT)),
)
def test_python_starts_from_each_source_directory(directory):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from types import MappingProxyType, GenericAlias; "
            "from logging import getLogger; "
            "from dataclasses import dataclass; "
            "from vicmf6.driver import run_coupling; "
            "from vicmf6.diagnostics import build_logger",
        ],
        cwd=directory,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(("", str(PACKAGE_ROOT.parent))),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Error processing line" not in result.stderr, result.stderr


@pytest.mark.parametrize(
    "package, old_name, new_name",
    [
        *[
            (package, "types", "records")
            for package in (
                "config",
                "coupling",
                "exchange",
                "mf6",
                "model_inputs",
                "vic",
                "post",
                "preprocess.exchange_builder",
            )
        ],
        ("diagnostics", "logging", "log_setup"),
    ],
)
def test_qualified_legacy_imports_use_the_renamed_module(package, old_name, new_name):
    from importlib import import_module

    legacy = import_module(f"vicmf6.{package}.{old_name}")
    current = import_module(f"vicmf6.{package}.{new_name}")
    assert legacy is current
    assert import_module(old_name) is not current
