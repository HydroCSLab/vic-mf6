"""start vicmf6 with shell-level python path overrides disabled."""

from __future__ import annotations

import os
import sys


def main() -> int:
    """re-exec the cli with the same interpreter and an isolated python path."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)

    # use the interpreter that owns this vicmf6 command, then ask python to
    # ignore inherited PYTHON* variables before scientific packages are loaded.
    arguments = [sys.executable, "-E", "-m", "vicmf6.cli", *sys.argv[1:]]
    os.execve(sys.executable, arguments, environment)
    raise RuntimeError("failed to replace the vicmf6 bootstrap process")


if __name__ == "__main__":
    raise SystemExit(main())
