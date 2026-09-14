#!/usr/bin/env bash

set -eu
if [ "$#" -ne 3 ]; then
    printf '%s\n' "usage: run_mf6_standalone.sh RANKS MF6_EXE WORKSPACE" >&2
    exit 2
fi
ranks=$1
exe=$2
workspace=$3
(
    cd "$workspace"
    mpirun -np "$ranks" "$exe"
)
