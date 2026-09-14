#!/usr/bin/env bash

# collect concise build/runtime provenance without changing the model stack.

set -u

printf '%s\n' "===== host ====="
uname -a || true
printf '%s\n' "===== python ====="
python3 --version || true
printf '%s\n' "===== mpi ====="
mpirun --version 2>/dev/null | head -5 || true
printf '%s\n' "===== compilers ====="
gcc --version 2>/dev/null | head -1 || true
gfortran --version 2>/dev/null | head -1 || true
printf '%s\n' "===== vicmf6 ====="
vicmf6 version 2>/dev/null || true
printf '%s\n' "===== python packages ====="
python3 - <<'PY' || true
from importlib.metadata import PackageNotFoundError, version
for name in ("numpy", "PyYAML", "mpi4py", "netCDF4", "xmipy"):
    try:
        print(f"{name}={version(name)}")
    except PackageNotFoundError:
        print(f"{name}=not-installed")
PY
printf '%s\n' "===== configured model paths ====="
printf 'VIC_SOURCE=%s\n' "${VIC_SOURCE:-}"
printf 'VIC_SAMPLE_DATA=%s\n' "${VIC_SAMPLE_DATA:-}"
printf 'MF6_LIB=%s\n' "${MF6_LIB:-}"
