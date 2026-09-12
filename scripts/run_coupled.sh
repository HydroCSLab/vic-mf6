#!/usr/bin/env bash

set -eu
if [ "$#" -ne 2 ]; then
    printf '%s\n' "usage: run_coupled.sh OUTER_RANKS CONFIG" >&2
    exit 2
fi
mpirun -np "$1" vicmf6 run -c "$2"
