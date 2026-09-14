#!/usr/bin/env bash

set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ranks=${1:-3}

if [ "$ranks" -ne 3 ]; then
    printf '%s\n' \
        "the deterministic MPI smoke fixture requires 3 ranks" >&2
    exit 2
fi

# use the python interpreter selected by the active environment. passing the
# absolute path through mpirun prevents worker ranks from finding a different
# system python through their own PATH lookup.
python_exe=${VICMF6_PYTHON:-$(command -v python)}

if [ -z "$python_exe" ] || [ ! -x "$python_exe" ]; then
    printf '%s\n' "python was not found in the active environment" >&2
    exit 2
fi

# -E prevents inherited PYTHONPATH/PYTHONHOME settings from injecting packages
# from another python installation into the MPI worker processes.
mpirun -np "$ranks" \
    "$python_exe" -E \
    "$repo_dir/tests/mpi_smoke.py"
