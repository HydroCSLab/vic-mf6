# Abd's VIC–MF6 run notes

Run these commands from this checkout. Choose **one** installation option.
Python 3.10+ is required. `pip` installs the Python framework; native coupling
also needs the modified VIC executable, parallel `libmf6.so`, and a compatible
MPI installation (`mpirun` and `mpicc`). See [native build instructions](bundle/docs/native-build.md)
if those are not installed, or use the Docker example below.

```bash
cd ~/projects/nmhydro/vic-mf6-submit
```

## 1. Install with a venv

```bash
python3 -E -m venv .venv
source .venv/bin/activate
export VICMF6_PYTHON="$VIRTUAL_ENV/bin/python"
"$VICMF6_PYTHON" -E -m pip install -e '.[runtime,preprocess,post,test]'
```

In later shells, repeat `source .venv/bin/activate` and the `export` line.

## Or install without a venv

Use a shell with no active venv:

```bash
export VICMF6_PYTHON="$(command -v python3)"
"$VICMF6_PYTHON" -E -m pip install --user -e '.[runtime,preprocess,post,test]'
```

Repeat the `export` line in later shells. If the system Python rejects a user
installation as externally managed, use the venv option.

Both options use an editable install: source edits take effect immediately.
The commands below use `./vicmf6`, which selects `VICMF6_PYTHON` and ignores
inherited Python path overrides. Keep that launcher in the repository; it needs
the adjacent `src/` directory.

## 2. Check installation and run unit tests

```bash
./vicmf6 version
./scripts/run_unit_tests.sh
```

Optional: exercise real MPI with small in-memory model substitutes:

```bash
VICMF6_RUN_MPI_TESTS=1 "$VICMF6_PYTHON" -E -m pytest tests/test_mpi_coupling.py -q
```

## 3. Run the complete native Stehekin example

```bash
cp examples/stehekin/config.example.yml examples/stehekin/config.local.yml
vi examples/stehekin/config.local.yml
```

Replace `@MF6_LIBRARY@`, `@VIC_GLOBAL_FILE@`, and `@VIC_EXECUTABLE@` with absolute
paths. The VIC global file must point to available Stehekin input data.
Then run:

```bash
./scripts/acceptance_stehekin.sh examples/stehekin/config.local.yml
cat examples/stehekin/run/postprocessing/acceptance_summary.txt
```

This builds the small MF6 fixture and MPI helper, generates the overlap table,
runs ten coupling windows, and writes tables, figures, and a report. It replaces
previous generated Stehekin products under `examples/stehekin/run/` and the
example's exchange table. Use this script for the disposable example only.

**Docker alternative:** builds the native components and includes the example
data; the host Python installation above is not needed for this route.

```bash
git submodule update --init --recursive
./bundle/scripts/build-image.sh
./bundle/scripts/run-acceptance.sh bundle/results/abd-stehekin-01
cat bundle/results/abd-stehekin-01/acceptance-status.txt
```

Choose a new or empty result directory each time. Rebuild the image after source
changes to test the updated code.

## 4. Couple my own VIC and MF6 models

Prepare a working copy of your model inputs. Each coupled GWF model in
`mfsim.nam` needs an API6 package. VIC and MF6 must cover the same time period,
MF6 time units must be days, and coupling boundaries must match the native
timesteps. Save MF6 heads and budgets for postprocessing.

First create the overlap table from your grids (replace paths and the CRS):

```bash
./native/build_disconnect.sh
./scripts/build_exchange_table.sh \
    /path/to/my-case/vic.global.txt \
    /path/to/my-case/mf6/mfsim.nam \
    /path/to/my-case/exchange_table.csv \
    EPSG:5070
```

For `pressure_head_from_interface_elevation`, fill the table's
`vic_interface_elevation_m` with each VIC cell's soil-base elevation in MF6's
vertical datum. A fifth script argument supplies one constant elevation only
when a uniform interface is appropriate for the model.

Create your configuration:

```bash
cp examples/stehekin/config.example.yml /path/to/my-case/config.yml
vi /path/to/my-case/config.yml
```

Set `mf6.namefile`, `mf6.library`, `vic.global_file`, `vic.executable`,
`vic.preload_library`, and `coupling.exchange_table` to the corresponding absolute
paths. The helper is `native/libvic_parent_disconnect.so` in this checkout.
Set `run.directory` to a fresh output location. Choose the coupling interval,
exchange length, conductivity scale, and head transform for your case; the
Stehekin values are example settings. See the [configuration reference](docs/runtime-and-configuration.md).

Inspect, validate, and run **one VIC model plus one MF6 GWF model**:

```bash
./vicmf6 inspect -c /path/to/my-case/config.yml
./vicmf6 preflight -c /path/to/my-case/config.yml
mpirun -np 2 ./vicmf6 run -c /path/to/my-case/config.yml
./vicmf6 post all -c /path/to/my-case/config.yml
```

If the MF6 simulation contains **two coupled GWF models**, use this launch instead:

```bash
mpirun -np 3 ./vicmf6 run -c /path/to/my-case/config.yml
```

The rule is **outer ranks = 1 controller + number of coupled GWF models**.
`vic.mpi_processes` controls the separately spawned VIC ranks. Two GWF models
must be declared together in the configured MF6 simulation.

The run writes `diagnostics/`, `vic/`, and, after postprocessing,
`postprocessing/` under `run.directory`. MF6 writes in its own configured
workspace. For another run, use fresh coupler output directories and a copied
MF6 workspace if you want to preserve its previous outputs; `--run-directory`
alone does not move MF6 output.
