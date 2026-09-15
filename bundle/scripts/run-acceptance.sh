#!/usr/bin/env bash

# run the bundled stehekin case and keep its reviewable products on the host.

set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
image=${VICMF6_IMAGE:-vic-mf6:local}
output_dir=${1:-"$repo_dir/results/stehekin"}

if [ "$#" -gt 1 ]; then
    printf '%s\n' "usage: $0 [OUTPUT_DIRECTORY]" >&2
    exit 2
fi
if [ -e "$output_dir" ] && [ ! -d "$output_dir" ]; then
    printf '%s\n' "output path is not a directory: $output_dir" >&2
    exit 2
fi

mkdir -p "$output_dir"
if [ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]; then
    printf '%s\n' "output directory must be empty: $output_dir" >&2
    exit 2
fi
output_dir=$(realpath "$output_dir")

docker run --rm --init --shm-size=1g \
    --volume "$output_dir:/results/stehekin" \
    "$image" acceptance /results/stehekin
