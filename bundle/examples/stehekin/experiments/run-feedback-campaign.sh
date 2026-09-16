#!/usr/bin/env bash

# Reproduce every process experiment reported in the EMS manuscript.
# Generated model products belong in the caller's output directory, not in Git.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    printf 'usage: %s OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi

mkdir -p -- "$1"
output_root=$(CDPATH= cd -- "$1" && pwd)
if [ -n "$(find "$output_root" -mindepth 1 -print -quit)" ]; then
    printf 'output directory must be empty: %s\n' "$output_root" >&2
    exit 2
fi

install_dir=${VICMF6_INSTALL_DIR:-/opt/vicmf6}
example_dir=${VICMF6_EXAMPLE_DIR:-$install_dir/examples/stehekin}
vic_exe=${VICMF6_VIC_EXE:-$install_dir/bin/vic_image.exe}
mf6_library=${VICMF6_MF6_LIBRARY:-$install_dir/lib/libmf6.so}
coupler_dir=${VICMF6_COUPLER_DIR:-$install_dir/src/vic-mf6}
python_bin=${VICMF6_PYTHON:-python}
runner=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/run-feedback-experiment.py
plotter=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/plot-feedback-experiments.py

for required in "$example_dir/input/domain_stehekin_20151028.nc" \
    "$example_dir/input/stehekin_parameters_20160327.nc" \
    "$example_dir/input/stehekin_forcings_10_days_1949.nc" \
    "$example_dir/stehekin.global.txt" "$vic_exe" "$mf6_library"; do
    if [ ! -e "$required" ]; then
        printf 'required bundle input is missing: %s\n' "$required" >&2
        exit 1
    fi
done

sample_dir=$(mktemp -d "${TMPDIR:-/tmp}/vicmf6-feedback-sample.XXXXXX")
cleanup() {
    rm -rf -- "$sample_dir"
}
trap cleanup EXIT

mkdir -p "$sample_dir/parameters" "$sample_dir/forcings"
cp "$example_dir/input/domain_stehekin_20151028.nc" \
    "$sample_dir/parameters/domain.stehekin.20151028.nc"
cp "$example_dir/input/stehekin_parameters_20160327.nc" \
    "$sample_dir/parameters/Stehekin_test_params_20160327.nc"
cp "$example_dir/input/stehekin_forcings_10_days_1949.nc" \
    "$sample_dir/forcings/Stehekin_image_test.forcings_10days.1949.nc"

# The runner executes VIC from the sample directory. Absolute DOMAIN and
# PARAMETERS paths make the same runner work with the bundle's compact input
# naming and with the original VIC sample-data layout.
sed \
    -e "s|^DOMAIN .*|DOMAIN $sample_dir/parameters/domain.stehekin.20151028.nc|" \
    -e "s|^PARAMETERS .*|PARAMETERS $sample_dir/parameters/Stehekin_test_params_20160327.nc|" \
    "$example_dir/stehekin.global.txt" > "$sample_dir/parameters/Stehekin_image_test.global.txt"

run_case() {
    local name=$1
    shift
    printf '[campaign] %s\n' "$name"
    "$python_bin" "$runner" \
        --run-dir "$output_root/$name" \
        --sample-dir "$sample_dir" \
        --vic-exe "$vic_exe" \
        --mf6-library "$mf6_library" \
        --coupler-dir "$coupler_dir" \
        --days 60 \
        "$@"
}

run_case baseline --case baseline --pumping-mm-day 1
run_case pumped --case pumped --pumping-mm-day 1
run_case tight-baseline --case baseline --aquitard-k 0.002 --pumping-mm-day 1
run_case tight-pumped --case pumped --aquitard-k 0.002 --pumping-mm-day 1
run_case pumped-3mm --case pumped --pumping-mm-day 3
run_case baseline-6h --case baseline --interval 0.25 --pumping-mm-day 1
run_case pumped-6h --case pumped --interval 0.25 --pumping-mm-day 1
run_case high-snow --case baseline --snowfall-mm-day 24 --pumping-mm-day 1
run_case stock --case stock --pumping-mm-day 1
run_case replay --case replay --pumping-mm-day 1 \
    --replay-from "$output_root/baseline"

mkdir -p "$output_root/analysis" "$output_root/figures"
"$python_bin" "$plotter" \
    --campaign-dir "$output_root" \
    --analysis-dir "$output_root/analysis" \
    --figure-dir "$output_root/figures"

printf '[OK] completed ten manuscript process experiments in %s\n' "$output_root"
