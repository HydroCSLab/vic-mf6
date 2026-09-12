#!/usr/bin/env bash

# run the source-tree tests with inherited shell python paths disabled.

set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_executable=${VICMF6_PYTHON:-$(command -v python)}

exec env -u PYTHONPATH -u PYTHONHOME \
    "$python_executable" -E -c '
import sys
import pytest

source_directory = sys.argv[1]
sys.path.insert(0, source_directory)
raise SystemExit(pytest.main(["-q"]))
' "$repo_dir/src"
