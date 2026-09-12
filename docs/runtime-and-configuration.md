# Runtime and configuration

## Runtime compatibility

`mpi4py`, VIC, MODFLOW 6, `libmf6.so`, and the MPI launcher must use a
compatible MPI implementation. Avoid changing the MPI provider inside an
otherwise validated environment.

The modified VIC Image Driver must accept the spatial groundwater-head input,
write signed `OUT_GW_EXCHANGE`, and support restart-linked coupling windows.
Each coupled MODFLOW 6 GWF model must contain an API6 package named `VICAPI`.
MODFLOW 6 time units must be days, and time steps must end at coupling
boundaries.

## Python environments

For an existing model and MPI stack:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[runtime,preprocess,post,test]'
```

The repository also provides a Conda environment:

```bash
mamba env create -f environment.yml
mamba activate vicmf6
python -m pip install -e .
```

## Configuration file

Paths relative to the YAML file are resolved from that file's directory.
Absolute paths are useful for external model builds. A minimal configuration
has this form:

```yaml
run:
  directory: run

mf6:
  namefile: run/mf6/mfsim.nam
  library: /absolute/path/to/libmf6.so
  max_solve_iterations: 100

vic:
  global_file: /absolute/path/to/vic.global.txt
  executable: /absolute/path/to/vic_image.exe
  mpi_processes: 1
  omp_threads: 1
  spawn_timeout_seconds: 3600
  preload_library: ../../native/libvic_parent_disconnect.so

coupling:
  exchange_table: exchange_table.csv
  interval_days: 1.0
  scheme: explicit
  exchange_length_m: 100.0
  exchange_conductivity_scale: 0.001
  head_transform: pressure_head_from_interface_elevation
  require_full_vic_coverage: true
  coverage_relative_tolerance: 1.0e-10
  conservation_absolute_tolerance_m3: 1.0e-5
  conservation_relative_tolerance: 1.0e-11
  api_absolute_tolerance_m3_per_day: 1.0e-5

diagnostics:
  verbosity: info
  write_rank_logs: true
```

Model dates, VIC cell metadata, GWF model names, MODFLOW 6 time steps, and API
package information are discovered from the model input files. They are not
duplicated in the coupling YAML.

## Inspect and preflight

Use the human-readable inspection before every new configuration:

```bash
vicmf6 inspect -c config.yml
```

Use the machine-readable preflight in automated workflows:

```bash
vicmf6 preflight -c config.yml > preflight.json
```

Preflight checks paths, model dates and schedules, GWF/API6 assignments,
exchange-table coverage, units, and MPI topology without advancing either
model.

## Exchange-table preprocessing

Build the conservative overlap table from the actual VIC and MODFLOW 6 grids:

```bash
./scripts/build_exchange_table.sh \
    /path/to/vic.global.txt \
    /path/to/mfsim.nam \
    /path/to/exchange_table.csv \
    EPSG:5070 \
    -1.0
```

The optional final argument supplies a constant interface elevation in metres.
A production application should use interface elevations in the same vertical
datum as MODFLOW 6. The command refuses to replace an existing output unless
`--force` appears before the positional arguments.

## Launch rules

If a simulation contains `N` coupled GWF models, launch exactly one controller
plus `N` groundwater workers:

```bash
./scripts/run_coupled.sh $((N + 1)) config.yml
```

The equivalent direct command is:

```bash
mpirun -np $((N + 1)) vicmf6 run -c config.yml
```

The `vic.mpi_processes` configuration controls the separately spawned VIC
child job. It does not change the required outer world size.

`--run-directory` relocates coupler-owned VIC and diagnostics outputs for one
launch. It does not relocate the configured MODFLOW 6 workspace. Use a copied
configuration and model workspace when an existing MF6 run must remain
untouched.

## Outputs and postprocessing

Coupler-owned products are written below the configured run directory:

```text
run/
+-- vic/                 per-window VIC inputs, exchange, outputs, and restarts
+-- diagnostics/         rank logs, window records, and run summary
+-- postprocessing/      derived tables, acceptance record, report, and figures
```

After a successful run:

```bash
vicmf6 post summarize -c config.yml
vicmf6 post accept -c config.yml
vicmf6 post figures -c config.yml
vicmf6 post all -c config.yml
```

`post all` produces the complete package. Postprocessing reads model outputs
and writes derived products; it does not change the completed VIC or MODFLOW 6
results.

## Routine checks and provenance

```bash
./scripts/run_unit_tests.sh
./scripts/run_mpi_smoke.sh 3
./scripts/run_mpi_failure_smoke.sh
./scripts/collect_provenance.sh > provenance.txt
```

Preserve the exact VIC and MODFLOW 6 revisions, local patches, compiler flags,
MPI implementation, Python environment, coupling configuration, exchange-table
summary, and acceptance outputs with published runs.
