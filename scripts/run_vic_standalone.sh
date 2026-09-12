#!/usr/bin/env bash

# run an unchanged vic global file from its own working directory.

set -eu

if [ "$#" -ne 3 ]; then
    printf '%s\n' "usage: run_vic_standalone.sh RANKS VIC_EXE GLOBAL_FILE" >&2
    exit 2
fi

ranks=$1
exe=$2
global=$3
workdir=$(CDPATH= cd -- "$(dirname -- "$global")" && pwd)
global_name=$(basename -- "$global")

(
    cd "$workdir"
    mpirun -np "$ranks" "$exe" -g "$global_name" </dev/null
)
