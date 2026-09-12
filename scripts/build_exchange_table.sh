#!/usr/bin/env bash

# Build a conservative VIC-MODFLOW 6 overlap table from the actual model grids.
#
# Usage:
#   build_exchange_table.sh [--force] VIC_GLOBAL MF6_MFSIM OUTPUT_CSV MF6_CRS [INTERFACE_ELEVATION_M]
#
# Example:
#   ./scripts/build_exchange_table.sh --force \
#       /path/to/vic.global.txt \
#       examples/stehekin/run/mf6/mfsim.nam \
#       examples/stehekin/exchange_table.csv \
#       EPSG:5070 \
#       -1.0

set -eu

force=false
if [ "${1:-}" = "--force" ]; then
    force=true
    shift
fi

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    printf '%s\n' \
        "usage: $0 [--force] VIC_GLOBAL MF6_MFSIM OUTPUT_CSV MF6_CRS [INTERFACE_ELEVATION_M]" >&2
    exit 2
fi

vic_global=$1
mf6_sim=$2
output=$3
mf6_crs=$4
interface_elevation=${5:-}

args=(
    --vic-global "$vic_global"
    --mf6-sim "$mf6_sim"
    --output "$output"
    --mf6-crs "$mf6_crs"
)

if [ "$force" = true ]; then
    args+=(--force)
fi

if [ -n "$interface_elevation" ]; then
    args+=(--interface-elevation-m "$interface_elevation")
fi

python -E -m vicmf6.preprocess.exchange_builder "${args[@]}"
