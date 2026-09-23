#!/usr/bin/env bash

# Run the bundle's Stehekin acceptance case with a developer-native install.
# The temporary working tree keeps generated files out of the Git checkout.

set -euo pipefail

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
project_dir=$(CDPATH= cd -- "$bundle_dir/.." && pwd)
prefix=${1:-}
output_dir=${2:-"$bundle_dir/results/native-stehekin"}

usage() {
    printf 'usage: %s INSTALL_PREFIX [OUTPUT_DIRECTORY]\n' "$0" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 2
fi
if [ ! -d "$prefix" ]; then
    printf 'install prefix is not a directory: %s\n' "$prefix" >&2
    exit 2
fi
prefix=$(realpath "$prefix")

python_exe="$prefix/venv/bin/python"
for required in \
    "$python_exe" \
    "$prefix/bin/vic_image.exe" \
    "$prefix/bin/mf6" \
    "$prefix/lib/libmf6.so" \
    "$prefix/lib/libvic_parent_disconnect.so"
do
    if [ ! -x "$required" ] && [ ! -f "$required" ]; then
        printf 'native install is incomplete; missing: %s\n' "$required" >&2
        exit 2
    fi
done

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

work_root=${VICMF6_ACCEPTANCE_WORK_DIR:-$output_dir/.work}
mkdir -p -- "$work_root"
export TMPDIR="$work_root"
export OMPI_MCA_orte_tmpdir_base="$work_root/ompi"
export MPLCONFIGDIR="$work_root/matplotlib"
mkdir -p -- "$OMPI_MCA_orte_tmpdir_base" "$MPLCONFIGDIR"
work_dir=$(mktemp -d "$work_root/vicmf6-native-acceptance.XXXXXX")
cleanup() {
    rm -rf -- "$work_dir"
}
trap cleanup EXIT

coupler_dir="$work_dir/vic-mf6"
example_dir="$coupler_dir/examples/stehekin"
mkdir -p "$coupler_dir"
cp "$project_dir/pyproject.toml" "$project_dir/README.md" "$project_dir/COPYING" "$project_dir/MANIFEST.in" "$coupler_dir/"
cp "$project_dir/vicmf6" "$coupler_dir/vicmf6"
cp -a "$project_dir/src" "$project_dir/native" "$project_dir/scripts" "$project_dir/examples" "$coupler_dir/"
cp -a "$bundle_dir/examples/stehekin/." "$example_dir/"

render_config() {
    "$python_exe" -E - "$example_dir/config.yml" \
        "$example_dir/config.local.yml" "$prefix" <<'PY'
from pathlib import Path
import sys

template = Path(sys.argv[1]).read_text(encoding="utf-8")
output = Path(sys.argv[2])
prefix = sys.argv[3]
marker = "@INSTALL_DIR@"

if template.count(marker) != 3:
    raise SystemExit(f"expected three {marker} markers in {output}")

output.write_text(template.replace(marker, prefix), encoding="utf-8")
PY
}

collect_artifacts() {
    local status=$1
    local artifact

    if [ -d "$example_dir/run" ]; then
        cp -a "$example_dir/run/." "$output_dir/"
    fi
    for artifact in exchange_table.csv exchange_table.summary.json exchange_table.summary.txt; do
        if [ -f "$example_dir/$artifact" ]; then
            cp "$example_dir/$artifact" "$output_dir/$artifact"
        fi
    done
    if [ "$status" -eq 0 ]; then
        printf 'PASS\n' > "$output_dir/acceptance-status.txt"
    else
        printf 'FAIL exit_code=%s\n' "$status" > "$output_dir/acceptance-status.txt"
    fi
    {
        cat "$bundle_dir/components.lock"
        printf '\n'
        "$python_exe" --version
        mpirun --version
        "$prefix/bin/mf6" -v
        "$python_exe" -m pip freeze
    } > "$output_dir/software-environment.txt"
}

run_step() {
    local status

    if "$@"; then
        return 0
    fi
    status=$?
    collect_artifacts "$status"
    printf '[FAIL] native Stehekin acceptance; diagnostics: %s\n' "$output_dir" >&2
    exit "$status"
}

export PATH="$prefix/venv/bin:$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VICMF6_PYTHON="$python_exe"
export MPLCONFIGDIR="$work_dir/matplotlib"
mkdir -p "$MPLCONFIGDIR"

run_step render_config
run_step "$python_exe" -E "$example_dir/create_mf6.py"
run_step "$coupler_dir/scripts/build_stehekin_exchange_table.sh" \
    "$example_dir/config.local.yml"
run_step "$coupler_dir/vicmf6" inspect -c "$example_dir/config.local.yml"
run_step mpirun -np 2 "$coupler_dir/vicmf6" run -c "$example_dir/config.local.yml"
run_step "$python_exe" -E "$coupler_dir/scripts/check_stehekin_result.py" \
    "$example_dir/run/diagnostics"
run_step "$coupler_dir/vicmf6" post all -c "$example_dir/config.local.yml"

collect_artifacts 0
printf '[OK] complete native Stehekin acceptance workflow\n'
printf 'results: %s\n' "$output_dir"
