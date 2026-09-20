#!/usr/bin/env bash

# Retain every manuscript experiment and diagnostic under the project analysis
# directory. An explicit first argument remains available for a separate run.
set -euo pipefail
get_project_dir() {
    repository_dir=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
    dirname -- "$repository_dir"
}

project_dir=$(get_project_dir)
output_dir=${VICMF6_MANUSCRIPT_OUTPUT_DIR:-$project_dir/analysis/vic-mf6-manuscript}
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    output_dir=$1
    shift
fi
if [[ "$output_dir" != /* ]]; then
    output_dir=$project_dir/$output_dir
fi
mkdir -p -- "$output_dir"
output_dir=$(realpath -- "$output_dir")
if [ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]; then
    printf 'output directory must be empty: %s\n' "$output_dir" >&2
    exit 2
fi
image=${VICMF6_IMAGE:-vic-mf6:manuscript}
printf 'Manuscript results: %s\n' "$output_dir"
printf 'Container image: %s\n' "$image"
printf 'Image ID: '
docker image inspect --format '{{.Id}}' "$image"
if [ -n "${VICMF6_COLOR:-}" ]; then
    color_setting=$VICMF6_COLOR
elif [ -t 1 ]; then
    color_setting=always
else
    color_setting=never
fi
if docker run --rm --init --shm-size=1g \
        --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
        --env MKL_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
        --env "VICMF6_COLOR=$color_setting" \
        --volume "$output_dir:/results/manuscript" \
        "$image" manuscript /results/manuscript "$@"
then
    printf '[OK] manuscript results saved on host: %s\n' "$output_dir"
else
    status=$?
    printf '[FAIL] manuscript run failed; partial results and logs are in: %s\n' "$output_dir" >&2
    exit "$status"
fi
