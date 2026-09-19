#!/usr/bin/env bash

# Retain every manuscript experiment and diagnostic in a new host directory.
set -euo pipefail
if [ "$#" -lt 1 ]; then
    printf 'usage: %s OUTPUT_DIRECTORY [--workers N] [--stages LIST]\n' "$0" >&2
    exit 2
fi
output_dir=$1
shift
mkdir -p -- "$output_dir"
output_dir=$(realpath -- "$output_dir")
if [ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]; then
    printf 'output directory must be empty: %s\n' "$output_dir" >&2
    exit 2
fi
image=${VICMF6_IMAGE:-vic-mf6:manuscript}
docker image inspect --format '{{.Id}}' "$image"
exec docker run --rm --init --shm-size=1g \
    --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
    --volume "$output_dir:/results/manuscript" \
    "$image" manuscript /results/manuscript "$@"
