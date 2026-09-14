#!/usr/bin/env bash

# Rebuild the disposable Stehekin overlap table from its configured model grids.

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
    exit 2
fi

config=$(realpath "$config")
if [ "$(dirname -- "$config")" != "$example_dir" ]; then
    printf '%s\n'         "the Stehekin configuration must be stored under $example_dir" >&2
    exit 2
fi

python_exe=${VICMF6_PYTHON:-$(command -v python)}
if [ -z "$python_exe" ] || [ ! -x "$python_exe" ]; then
    printf '%s\n' "python was not found in the active environment" >&2
    exit 2
fi

mapfile -t resolved < <(
    "$python_exe" -E - "$config" <<'PY'
from pathlib import Path
import sys

import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
base = config_path.parent


def resolve(value):
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


print(resolve(data["vic"]["global_file"]))
print(resolve(data["mf6"]["namefile"]))
print(resolve(data["coupling"]["exchange_table"]))
print(str(data["coupling"].get("head_transform", "")))
PY
)

if [ "${#resolved[@]}" -ne 4 ]; then
    printf '%s\n' "failed to resolve Stehekin paths from $config" >&2
    exit 2
fi

vic_global=${resolved[0]}
mf6_sim=${resolved[1]}
output=${resolved[2]}
head_transform=${resolved[3]}

expected_mf6="$example_dir/run/mf6/mfsim.nam"
expected_output="$example_dir/exchange_table.csv"
if [ "$mf6_sim" != "$expected_mf6" ]; then
    printf '%s\n' "mf6.namefile must resolve to $expected_mf6" >&2
    exit 2
fi
if [ "$output" != "$expected_output" ]; then
    printf '%s\n' "coupling.exchange_table must resolve to $expected_output" >&2
    exit 2
fi
if [ "$head_transform" != "pressure_head_from_interface_elevation" ]; then
    printf '%s\n'         "coupling.head_transform must be pressure_head_from_interface_elevation" >&2
    exit 2
fi

cd "$repo_dir"
exec ./scripts/build_exchange_table.sh --force     "$vic_global"     "$mf6_sim"     "$output"     EPSG:5070     -1.0
