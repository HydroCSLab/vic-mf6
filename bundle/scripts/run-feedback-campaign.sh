#!/usr/bin/env bash

# Run the complete manuscript process campaign in the tested container image.
# All generated inputs, model products, audits, and figures stay on the host
# below the caller-provided empty output directory.

set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
image=${VICMF6_IMAGE:-vic-mf6:local}
output_dir=${1:-"$repo_dir/results/manuscript-campaign"}

if [ "$#" -gt 1 ]; then
    printf 'usage: %s [OUTPUT_DIRECTORY]\n' "$0" >&2
    exit 2
fi
if [ -e "$output_dir" ] && [ ! -d "$output_dir" ]; then
    printf 'output path is not a directory: %s\n' "$output_dir" >&2
    exit 2
fi

mkdir -p "$output_dir"
if [ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]; then
    printf 'output directory must be empty: %s\n' "$output_dir" >&2
    exit 2
fi
output_dir=$(realpath "$output_dir")

exec docker run --rm --init --shm-size=1g \
    --volume "$output_dir:/results/campaign" \
    "$image" feedback-campaign /results/campaign
