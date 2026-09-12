#!/usr/bin/env bash

set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output="$here/libvic_parent_disconnect.so"

command -v mpicc >/dev/null 2>&1 || {
    printf '%s\n' "mpicc was not found on PATH" >&2
    exit 2
}

mpicc -shared -fPIC -O2 -Wall -Wextra \
    -o "$output" "$here/mpi_finalize_disconnect.c"

printf '[OK] built %s\n' "$output"
