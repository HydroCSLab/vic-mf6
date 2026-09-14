#!/usr/bin/env bash

set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_exe=${VICMF6_PYTHON:-$(command -v python)}
output_file=$(mktemp "${TMPDIR:-/tmp}/vicmf6-mpi-failure.XXXXXX")
trap 'rm -f "$output_file"' EXIT

set +e
mpirun -np 3 "$python_exe" -E \
    "$repo_dir/tests/mpi_failure_smoke.py" >"$output_file" 2>&1
status=$?
set -e

cat "$output_file"
if [ "$status" -eq 0 ]; then
    printf '%s\n' "[FAIL] intentional MPI worker failure returned success" >&2
    exit 1
fi

if ! grep -Fq '[intentional-worker-failure]' "$output_file"; then
    printf '%s\n' \
        "[FAIL] MPI failed before the intentional worker-abort marker was emitted" >&2
    exit 1
fi

printf '[OK] intentional MPI worker failure propagated with status %s\n' "$status"
