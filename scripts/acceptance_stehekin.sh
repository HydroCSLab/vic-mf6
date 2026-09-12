#!/usr/bin/env bash

# Run the complete disposable Stehekin acceptance case.

set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
example_dir="$repo_dir/examples/stehekin"
config=${1:-"$example_dir/config.local.yml"}

if [ "$#" -gt 1 ]; then
    printf '%s\n' "usage: $0 [CONFIG]" >&2
    exit 2
fi
if [ ! -f "$config" ]; then
    printf '%s\n' "Stehekin configuration was not found: $config" >&2
    printf '%s\n'         "copy examples/stehekin/config.example.yml to config.local.yml and set the three CHANGE_ME paths" >&2
    exit 2
fi

config=$(realpath "$config")
if [ "$(dirname -- "$config")" != "$example_dir" ]; then
    printf '%s\n'         "the acceptance configuration must be stored under $example_dir" >&2
    exit 2
fi

python_exe=${VICMF6_PYTHON:-$(command -v python)}
if [ -z "$python_exe" ] || [ ! -x "$python_exe" ]; then
    printf '%s\n' "python was not found in the active environment" >&2
    exit 2
fi
if ! command -v mpirun >/dev/null 2>&1; then
    printf '%s\n' "mpirun was not found on PATH" >&2
    exit 2
fi

run_dir="$example_dir/run"
diagnostics="$run_dir/diagnostics"

printf '%s\n' "[INFO] replacing generated Stehekin products under $run_dir"
rm -rf -- "$diagnostics" "$run_dir/mf6" "$run_dir/vic" "$run_dir/postprocessing"

"$python_exe" -E "$example_dir/build_mf6.py"
"$repo_dir/scripts/build_stehekin_exchange_table.sh" "$config"
"$repo_dir/native/build_disconnect.sh"
"$repo_dir/vicmf6" inspect -c "$config"
mpirun -np 2 "$repo_dir/vicmf6" run -c "$config"
"$python_exe" -E "$repo_dir/scripts/check_stehekin_result.py" "$diagnostics"
"$repo_dir/vicmf6" post all -c "$config"

printf '%s\n' "[OK] complete Stehekin acceptance workflow"
printf '%s\n' "diagnostics: $diagnostics"
printf '%s\n' "report: $run_dir/postprocessing/report.md"
