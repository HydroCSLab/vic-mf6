# Native developer build

This is the advanced Linux workflow for developers who need to change, debug,
or profile a pinned component outside Docker.
For a reproducible review run, use the Docker workflow in the top-level README.
The commands below intentionally mirror the compiler, MPI, and package choices
in the bundle Dockerfile.

## Scope and requirements

The native route is validated only on 64-bit Linux.
All three components must use the same MPI implementation and compatible
compiler toolchain.
Mixing a system MPI library with a component built against another MPI library
can fail at startup or corrupt MPI communication.

On Ubuntu 24.04, install the build dependencies with:

```bash
sudo apt-get update
sudo apt-get install --yes \
    build-essential gfortran pkg-config \
    libnetcdf-dev libnetcdff-dev \
    libopenmpi-dev openmpi-bin libpetsc-real-dev \
    meson ninja-build python3 python3-dev python3-pip python3-venv
```

Clone the bundle recursively and verify its exact source lock before building:

```bash
git clone --recurse-submodules https://github.com/mabdazzam/vic-mf6.git
cd vic-mf6
./bundle/scripts/check-components.sh
```

The commands below install only under `.native/`, which is ignored by Git.
Choose another absolute prefix if you maintain a shared development install.

```bash
export BUNDLE_DIR="$PWD/bundle"
export VICMF6_PREFIX="$PWD/.native/install"
export VICMF6_BUILD="$PWD/.native/build"
mkdir -p "$VICMF6_PREFIX/bin" "$VICMF6_PREFIX/lib" "$VICMF6_BUILD"
```

## Build VIC Image Driver

Build the modified VIC source with the NetCDF and MPI development libraries
visible on the active `PATH` and compiler search path:

```bash
vic_revision=$(git -C "$BUNDLE_DIR/components/vic" rev-parse HEAD)
make -C "$BUNDLE_DIR/components/vic/vic/drivers/image" model \
    GIT_VERSION="$vic_revision" \
    HOSTNAME=vicmf6-native \
    LOG_LVL=30 \
    USER="$USER"
install -D -m 0755 "$BUNDLE_DIR/components/vic/vic/drivers/image/vic_image.exe" \
    "$VICMF6_PREFIX/bin/vic_image.exe"
```

The VIC checkout includes the coupling-specific source changes.
Do not substitute an arbitrary released VIC executable: it must read the
groundwater-head interface and write signed `OUT_GW_EXCHANGE` output.

## Build parallel MODFLOW 6 and its XMI library

Use the same OpenMPI Fortran wrapper to configure and link MODFLOW 6.
The parallel option builds both the `mf6` executable and `libmf6.so` used by
the coupling API:

```bash
FC=mpifort meson setup "$VICMF6_BUILD/modflow6" "$BUNDLE_DIR/components/modflow6" \
    --buildtype=release \
    --prefix="$VICMF6_PREFIX" \
    --bindir=bin \
    --libdir=lib \
    -Dparallel=true \
    -Dfortran_args=-ffree-line-length-none
meson compile -C "$VICMF6_BUILD/modflow6"
meson install -C "$VICMF6_BUILD/modflow6"
```

`-ffree-line-length-none` makes the source compatible with current GNU
Fortran compilers, which otherwise reject a few standards-conforming long
source lines under strict Fortran 2018 diagnostics.
When reconfiguring an existing build directory, replace the first command with
the same command plus `--reconfigure`.

## Install vicmf6 and the MPI helper

Create an isolated Python environment and compile `mpi4py` against the active
MPI installation.
The requirement file is deliberately pinned to the image's scientific Python
stack so the experiment scripts have FloPy and Matplotlib available:

```bash
python3 -m venv "$VICMF6_PREFIX/venv"
. "$VICMF6_PREFIX/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
MPICC=mpicc python -m pip install --ignore-installed --no-binary=mpi4py \
    --requirement "$BUNDLE_DIR/requirements-container.txt"
python -m pip install --no-build-isolation --no-deps .
python -m pip install 'pytest>=8'

mpicc -shared -fPIC -O2 -Wall -Wextra \
    -o "$VICMF6_PREFIX/lib/libvic_parent_disconnect.so" \
    native/mpi_finalize_disconnect.c
```

Set the runtime paths for the current shell:

```bash
export PATH="$VICMF6_PREFIX/venv/bin:$VICMF6_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$VICMF6_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify the installed programs before running a coupled case:

```bash
vic_image.exe -v
mf6 -v
vicmf6 --help
python -m pytest tests -ra
```

## Native Stehekin acceptance

The bundle wrapper creates a temporary coupled working tree from the pinned
coupler source and the public Stehekin sample input.
It renders a local configuration from the install prefix, runs the same
two-rank acceptance path as the image, and writes only review products to the
requested output directory:

```bash
cd /path/to/vic-mf6
./bundle/scripts/run-native-acceptance.sh "$VICMF6_PREFIX" bundle/results/native-stehekin
```

The output layout and acceptance checks match the Docker workflow.
The component README remains the reference for direct operator commands and
for local configurations used while changing the coupler itself.

The native workflow is for development.
Before publishing a result or release image, repeat the Docker acceptance
workflow from a clean recursive bundle checkout.
