#!/usr/bin/env bash

# create a local python environment for an already installed mpi/model stack.

set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv_dir=${1:-"$repo_dir/.venv"}

python3 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -e "$repo_dir[runtime,test]"

printf '%s\n' "[OK] python environment created: $venv_dir"
printf '%s\n' "activate with: . $venv_dir/bin/activate"
